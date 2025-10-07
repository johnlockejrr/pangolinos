#
# Copyright 2025 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
pangolinos.polygonize
~~~~~~~~~~~~~~~~~~~~

Polygonization functionality for text lines using the complete kraken algorithm.
"""
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, distance_transform_cdt, binary_erosion
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from skimage.filters import sobel
from skimage import draw
from skimage.transform import AffineTransform
from skimage.measure import approximate_polygon
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import explain_validity
import logging
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

# Set up logging
logger = logging.getLogger(__name__)

# Kraken constants
SEGMENTATION_HYPER_PARAMS = {'line_width': 8}


class SimpleXMLPage:
    """Simplified XMLPage class for PAGE-XML and ALTO parsing."""
    
    def __init__(self, filename):
        self.filename = filename
        self.imagename = None
        self.lines = []
        self._parse_page()
    
    def _parse_page(self):
        """Parse PAGE-XML or ALTO file and extract lines and image."""
        with open(self.filename, 'rb') as fp:
            doc = etree.parse(fp)
        
        # Check if it's ALTO format
        if doc.getroot().tag.endswith('alto'):
            self._parse_alto(doc)
        else:
            self._parse_pagexml(doc)
    
    def _parse_alto(self, doc):
        """Parse ALTO format."""
        # Get image filename
        filename_elem = doc.find('.//{http://www.loc.gov/standards/alto/ns-v4#}fileName')
        if filename_elem is not None:
            self.imagename = filename_elem.text
        
        # Extract text lines
        lines = doc.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}TextLine')
        for line in lines:
            baseline = line.get('BASELINE')
            shape = line.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Polygon')
            
            if baseline and shape is not None:
                baseline_points = self._parse_coords(baseline)
                coords_points = self._parse_coords(shape.get('POINTS', ''))
                
                if baseline_points and coords_points:
                    self.lines.append({
                        'baseline': baseline_points,
                        'coords': coords_points
                    })
    
    def _parse_pagexml(self, doc):
        """Parse PAGE-XML format."""
        # Get image filename
        page = doc.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page')
        if page is not None:
            self.imagename = page.get('imageFilename')
        
        # Extract text lines
        lines = doc.findall('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextLine')
        for line in lines:
            baseline = line.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Baseline')
            coords = line.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Coords')
            
            if baseline is not None and coords is not None:
                baseline_points = self._parse_coords(baseline.get('points', ''))
                coords_points = self._parse_coords(coords.get('points', ''))
                
                if baseline_points and coords_points:
                    self.lines.append({
                        'baseline': baseline_points,
                        'coords': coords_points
                    })
    
    def _parse_coords(self, coords_str):
        """Parse coordinate string into list of points."""
        if not coords_str or coords_str.isspace():
            return []
        
        points = []
        coords = coords_str.split()
        for coord in coords:
            if ',' in coord:
                x, y = coord.split(',')
                points.append((int(float(x)), int(float(y))))
        return points
    
    def to_container(self):
        """Return self for compatibility."""
        return self


def _ray_intersect_boundaries(ray, direction, aabb):
    """
    Simplified version of [0] for 2d and AABB anchored at (0,0).
    [0] http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    """
    dir_fraction = np.empty(2, dtype=ray.dtype)
    dir_fraction[direction == 0.0] = np.inf
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (-ray[0]) * dir_fraction[0]
    t2 = (aabb[0] - ray[0]) * dir_fraction[0]
    t3 = (-ray[1]) * dir_fraction[1]
    t4 = (aabb[1] - ray[1]) * dir_fraction[1]

    tmin = max(min(t1, t2), min(t3, t4))
    tmax = min(max(t1, t2), max(t3, t4))

    t = min(x for x in [tmin, tmax] if x >= 0)
    return ray + (direction * t)


def _rotate(image, angle, center, scale, cval=0, order=0, use_skimage_warp=False):
    """
    Rotate an image at an angle with optional scaling
    """
    if isinstance(image, Image.Image):
        rows, cols = image.height, image.width
    else:
        rows, cols = image.shape[:2]
        assert len(image.shape) == 3 or len(image.shape) == 2, 'Image must be 2D or 3D'

    tform = AffineTransform(rotation=angle, scale=(1/scale, 1))
    corners = np.array([
        [0, 0],
        [0, rows - 1],
        [cols - 1, rows - 1],
        [cols - 1, 0]
    ])
    corners = tform.inverse(corners)
    minc = corners[:, 0].min()
    minr = corners[:, 1].min()
    maxc = corners[:, 0].max()
    maxr = corners[:, 1].max()
    out_rows = maxr - minr + 1
    out_cols = maxc - minc + 1
    output_shape = tuple(int(o) for o in np.around((out_rows, out_cols)))
    # fit output image in new shape
    translation = tform([[minc, minr]])
    tform = AffineTransform(rotation=angle, scale=(1/scale, 1), translation=[f for f in translation.flatten()])

    if isinstance(image, Image.Image):
        # PIL is much faster than scipy
        pdata = tform.params.flatten().tolist()[:6]
        resample = {0: Image.Resampling.NEAREST, 1: Image.Resampling.BILINEAR, 2: Image.Resampling.BICUBIC, 3: Image.Resampling.BICUBIC}.get(order, Image.Resampling.NEAREST)
        rotated = image.transform(output_shape, Image.Transform.AFFINE, pdata, resample=resample, fillcolor=cval)
    else:
        # use scipy for numpy arrays
        from skimage.transform import warp
        rotated = warp(image, tform.inverse, output_shape=output_shape, cval=cval, order=order)

    return tform, rotated


def make_polygonal_mask(polygon, shape):
    """
    Creates a mask from a polygon.
    """
    mask = Image.new('L', shape, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon.astype(int).tolist()], fill=255, width=2)
    return mask


def _calc_roi(line, bounds, baselines, suppl_obj, p_dir):
    # interpolate baseline
    ls = LineString(line)
    ip_line = [line[0]]
    dist = 10
    while dist < ls.length:
        ip_line.append(np.array(ls.interpolate(dist).coords[0]))
        dist += 10
    ip_line.append(line[-1])
    ip_line = np.array(ip_line)
    upper_bounds_intersects = []
    bottom_bounds_intersects = []
    for point in ip_line:
        upper_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(-1, 1))[::-1], bounds+1).astype('int'))
        bottom_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(1, -1))[::-1], bounds+1).astype('int'))
    # build polygon between baseline and bbox intersects
    upper_polygon = Polygon(ip_line.tolist() + upper_bounds_intersects)
    bottom_polygon = Polygon(ip_line.tolist() + bottom_bounds_intersects)

    # select baselines at least partially in each polygon
    side_a = [LineString(upper_bounds_intersects)]
    side_b = [LineString(bottom_bounds_intersects)]

    for adj_line in baselines + suppl_obj:
        adj_line = LineString(adj_line)
        if upper_polygon.intersects(adj_line):
            side_a.append(adj_line)
        elif bottom_polygon.intersects(adj_line):
            side_b.append(adj_line)
    side_a = unary_union(side_a).buffer(1).boundary
    side_b = unary_union(side_b).buffer(1).boundary

    def _find_closest_point(pt, intersects):
        spt = LineString([pt, pt])  # Create a point
        if intersects.is_empty:
            raise Exception(f'No intersection with boundaries. Shapely intersection object: {intersects.wkt}')
        if intersects.geom_type == 'MultiPoint':
            return min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
        elif intersects.geom_type == 'Point':
            return intersects
        elif intersects.geom_type == 'GeometryCollection' and len(intersects.geoms) > 0:
            t = min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
            if t.geom_type == 'Point':
                return t
            else:
                return nearest_points(spt, t)[1]
        else:
            raise Exception(f'No intersection with boundaries. Shapely intersection object: {intersects.wkt}')

    env_up = []
    env_bottom = []
    # find orthogonal (to linear regression) intersects with adjacent objects to complete roi
    for point, upper_bounds_intersect, bottom_bounds_intersect in zip(ip_line, upper_bounds_intersects, bottom_bounds_intersects):
        upper_limit = _find_closest_point(point, LineString([point, upper_bounds_intersect]).intersection(side_a))
        bottom_limit = _find_closest_point(point, LineString([point, bottom_bounds_intersect]).intersection(side_b))
        env_up.append(upper_limit.coords[0])
        env_bottom.append(bottom_limit.coords[0])
    env_up = np.array(env_up, dtype='uint')
    env_bottom = np.array(env_bottom, dtype='uint')
    return env_up, env_bottom


def _calc_seam(baseline, polygon, angle, im_feats, bias=150):
    """
    Calculates seam between baseline and ROI boundary on one side.
    """
    MASK_VAL = 99999
    c_min, c_max = int(polygon[:, 0].min()), int(polygon[:, 0].max())
    r_min, r_max = int(polygon[:, 1].min()), int(polygon[:, 1].max())
    patch = im_feats[r_min:r_max+2, c_min:c_max+2].copy()
    # bias feature matrix by distance from baseline
    mask = np.ones_like(patch)
    for line_seg in zip(baseline[:-1] - (c_min, r_min), baseline[1:] - (c_min, r_min)):
        line_locs = draw.line(line_seg[0][1],
                              line_seg[0][0],
                              line_seg[1][1],
                              line_seg[1][0])
        mask[line_locs] = 0
    dist_bias = distance_transform_cdt(mask)
    # absolute mask
    mask = np.array(make_polygonal_mask(polygon-(c_min, r_min), patch.shape[::-1])) <= 128
    # dilate mask to compensate for aliasing during rotation
    mask = binary_erosion(mask, border_value=True, iterations=2)
    # combine weights with features
    patch[mask] = MASK_VAL
    patch += (dist_bias*(np.mean(patch[patch != MASK_VAL])/bias))
    extrema = baseline[(0, -1), :] - (c_min, r_min)
    # scale line image to max 600 pixel width
    scale = min(1.0, 600/(c_max-c_min))
    tform, rotated_patch = _rotate(patch,
                                   angle,
                                   center=extrema[0],
                                   scale=scale,
                                   cval=MASK_VAL,
                                   use_skimage_warp=True)
    # ensure to cut off padding after rotation
    x_offsets = np.sort(np.around(tform.inverse(extrema)[:, 0]).astype('int'))
    rotated_patch = rotated_patch[:, x_offsets[0]:x_offsets[1]+1]
    # infinity pad for seamcarve
    rotated_patch = np.pad(rotated_patch, ((1, 1), (0, 0)),  mode='constant', constant_values=np.inf)
    r, c = rotated_patch.shape
    # fold into shape (c, r-2 3)
    A = np.lib.stride_tricks.as_strided(rotated_patch, (c, r-2, 3), (rotated_patch.strides[1],
                                                                     rotated_patch.strides[0],
                                                                     rotated_patch.strides[0]))
    B = rotated_patch[1:-1, 1:].swapaxes(0, 1)
    backtrack = np.zeros_like(B, dtype='int')
    T = np.empty((B.shape[1]), 'f')
    R = np.arange(-1, len(T)-1)
    for i in np.arange(c-1):
        A[i].min(1, T)
        backtrack[i] = A[i].argmin(1) + R
        B[i] += T
    # backtrack
    seam = []
    j = np.argmin(rotated_patch[1:-1, -1])
    for i in range(c-2, -2, -1):
        seam.append((i+x_offsets[0]+1, j))
        j = backtrack[i, j]
    seam = np.array(seam)[::-1]
    seam_mean = seam[:, 1].mean()
    seam_std = seam[:, 1].std()
    seam[:, 1] = np.clip(seam[:, 1], seam_mean-seam_std, seam_mean+seam_std)
    # rotate back
    seam = tform(seam).astype('int')
    # filter out seam points in masked area of original patch/in padding
    seam = seam[seam.min(axis=1) >= 0, :]
    m = (seam < mask.shape[::-1]).T
    seam = seam[np.logical_and(m[0], m[1]), :]
    seam = seam[np.invert(mask[seam.T[1], seam.T[0]])]
    seam += (c_min, r_min)
    return seam


def _extract_patch(env_up, env_bottom, baseline, offset_baseline, end_points, dir_vec, topline, offset, im_feats, bounds):
    """
    Calculate a line image patch from a ROI and the original baseline.
    """
    upper_polygon = np.concatenate((baseline, env_up[::-1]))
    bottom_polygon = np.concatenate((baseline, env_bottom[::-1]))
    upper_offset_polygon = np.concatenate((offset_baseline, env_up[::-1]))
    bottom_offset_polygon = np.concatenate((offset_baseline, env_bottom[::-1]))

    angle = np.arctan2(dir_vec[1], dir_vec[0])
    roi_polygon = unary_union([Polygon(upper_polygon), Polygon(bottom_polygon)])

    if topline:
        upper_seam = _calc_seam(baseline, upper_polygon, angle, im_feats)
        bottom_seam = _calc_seam(offset_baseline, bottom_offset_polygon, angle, im_feats)
    else:
        upper_seam = _calc_seam(offset_baseline, upper_offset_polygon, angle, im_feats)
        bottom_seam = _calc_seam(baseline, bottom_polygon, angle, im_feats)

    upper_seam = LineString(upper_seam).simplify(5)
    bottom_seam = LineString(bottom_seam).simplify(5)

    # ugly workaround against GEOM parallel_offset bug creating a
    # MultiLineString out of offset LineString
    if upper_seam.parallel_offset(offset//2, side='right').geom_type == 'MultiLineString' or offset == 0:
        upper_seam = np.array(upper_seam.coords, dtype=int)
    else:
        upper_seam = np.array(upper_seam.parallel_offset(offset//2, side='right').coords, dtype=int)[::-1]
    if bottom_seam.parallel_offset(offset//2, side='left').geom_type == 'MultiLineString' or offset == 0:
        bottom_seam = np.array(bottom_seam.coords, dtype=int)
    else:
        bottom_seam = np.array(bottom_seam.parallel_offset(offset//2, side='left').coords, dtype=int)

    # offsetting might produce bounds outside the image. Clip it to the image bounds.
    polygon = np.concatenate(([end_points[0]], upper_seam, [end_points[-1]], bottom_seam[::-1]))
    polygon = Polygon(polygon)
    if not polygon.is_valid:
        polygon = np.concatenate(([end_points[-1]], upper_seam, [end_points[0]], bottom_seam))
        polygon = Polygon(polygon)
    if not polygon.is_valid:
        raise Exception(f'Invalid bounding polygon computed: {explain_validity(polygon)}')
    polygon = np.array(roi_polygon.intersection(polygon).boundary.coords, dtype=int)
    return polygon


def calculate_polygonal_environment(im, baselines, suppl_obj=None, im_feats=None, scale=None, topline=False, raise_on_error=False):
    """
    Given a list of baselines and an input image, calculates a polygonal
    environment around each baseline.
    """
    if scale is not None and (scale[0] > 0 or scale[1] > 0):
        w, h = im.size
        oh, ow = scale
        if oh == 0:
            oh = int(h * ow/w)
        elif ow == 0:
            ow = int(w * oh/h)
        im = im.resize((ow, oh))
        scale = np.array((ow/w, oh/h))
        # rescale baselines
        baselines = [(np.array(bl) * scale).astype('int').tolist() for bl in baselines]
        # rescale suppl_obj
        if suppl_obj is not None:
            suppl_obj = [(np.array(bl) * scale).astype('int').tolist() for bl in suppl_obj]

    if im_feats is None:
        bounds = np.array(im.size, dtype=float) - 1
        im = np.array(im.convert('L'))
        # compute image gradient
        im_feats = gaussian_filter(sobel(im), 0.5)
    else:
        bounds = np.array(im_feats.shape[::-1], dtype=float) - 1

    polygons = []
    if suppl_obj is None:
        suppl_obj = []

    for idx, line in enumerate(baselines):
        try:
            end_points = (line[0], line[-1])
            line = LineString(line)
            offset = SEGMENTATION_HYPER_PARAMS['line_width'] if topline is not None else 0
            offset_line = line.parallel_offset(offset, side='left' if topline else 'right')
            line = np.array(line.coords, dtype=float)
            offset_line = np.array(offset_line.coords, dtype=float)

            # calculate magnitude-weighted average direction vector
            lengths = np.linalg.norm(np.diff(line.T), axis=0)
            p_dir = np.mean(np.diff(line.T) * lengths/lengths.sum(), axis=1)
            p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
            env_up, env_bottom = _calc_roi(line, bounds, baselines[:idx] + baselines[idx+1:], suppl_obj, p_dir)

            polygons.append(_extract_patch(env_up,
                                           env_bottom,
                                           line.astype('int'),
                                           offset_line.astype('int'),
                                           end_points,
                                           p_dir,
                                           topline,
                                           offset,
                                           im_feats,
                                           bounds))
        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(f'Polygonizer failed on line {idx}: {e}')
            polygons.append(None)

    if scale is not None:
        polygons = [(np.array(pol)/scale).astype('uint').tolist() if pol is not None else None for pol in polygons]
    return polygons


def polygonize_document(doc: Union[str, 'PathLike'],
                        output_base_path: Union[str, 'PathLike'],
                        format_type: str = 'page',
                        topline: Optional[str] = None,
                        scale: int = 1800):
    """
    Polygonizes text lines in PAGE-XML or ALTO files using the complete kraken algorithm.
    
    Args:
        doc: Input XML file (PAGE-XML or ALTO)
        output_base_path: Directory to write output XML file into
        format_type: Input document format ('alto' or 'page')
        topline: Switch for baseline location ('topline', 'centerline', or 'baseline')
        scale: Integer height containing optional scale factors of the input
    """
    output_base_path = Path(output_base_path)
    doc = Path(doc)
    
    # Handle topline parameter
    if topline == 'topline':
        topline_flag = True
    elif topline == 'centerline':
        topline_flag = None
    else:  # baseline or default
        topline_flag = False

    # Parse the XML file
    seg = SimpleXMLPage(doc)
    if not seg.imagename:
        logger.warning(f'No image found for {doc}')
        return
        
    # Load image - try to find it in the same directory as the XML file
    import os
    xml_dir = os.path.dirname(seg.filename)
    image_path = os.path.join(xml_dir, seg.imagename)
    
    if not os.path.exists(image_path):
        logger.warning(f'Image not found at {image_path}')
        return
        
    im = Image.open(image_path).convert('L')
    
    # Extract baselines
    baselines = []
    for line in seg.lines:
        if line['baseline']:
            baselines.append(line['baseline'])
        else:
            baselines.append([])
    
    # Calculate polygons using exact kraken algorithm
    polygons = calculate_polygonal_environment(im, baselines, scale=(scale, 0), topline=topline_flag)
    
    # Replace in XML
    _replace_polygons_in_xml(doc, polygons, output_base_path)


def _replace_polygons_in_xml(doc_path, polygons, output_base_path):
    """Replace polygon coordinates in PAGE-XML or ALTO file."""
    with open(doc_path, 'rb') as fp:
        doc = etree.parse(fp)
    
    # Check if it's ALTO format
    if doc.getroot().tag.endswith('alto'):
        _replace_alto_polygons(doc, polygons, doc_path, output_base_path)
    else:
        _replace_pagexml_polygons(doc, polygons, doc_path, output_base_path)


def _replace_alto_polygons(doc, polygons, doc_path, output_base_path):
    """Replace polygon coordinates in ALTO file."""
    lines = doc.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}TextLine')
    idx = 0
    for line in lines:
        if line.get('BASELINE'):
            shape = line.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Polygon')
            if shape is not None:
                if idx < len(polygons) and polygons[idx] is not None:
                    shape.attrib['POINTS'] = ' '.join([','.join([str(x) for x in pt]) for pt in polygons[idx]])
                else:
                    shape.attrib['POINTS'] = ''
        idx += 1
    
    output_file = output_base_path / doc_path.name.replace('.xml', '_polygonized.xml')
    with open(output_file, 'wb') as fp:
        doc.write(fp, encoding='UTF-8', xml_declaration=True)


def _replace_pagexml_polygons(doc, polygons, doc_path, output_base_path):
    """Replace polygon coordinates in PAGE-XML file."""
    lines = doc.findall('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextLine')
    idx = 0
    for line in lines:
        base = line.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Baseline')
        if base is not None and base.get('points') and not base.get('points').isspace():
            pol = line.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Coords')
            if pol is not None:
                if idx < len(polygons) and polygons[idx] is not None:
                    pol.attrib['points'] = ' '.join([','.join([str(x) for x in pt]) for pt in polygons[idx]])
                else:
                    pol.attrib['points'] = ''
        idx += 1
    
    output_file = output_base_path / doc_path.name.replace('.xml', '_polygonized.xml')
    with open(output_file, 'wb') as fp:
        doc.write(fp, encoding='UTF-8', xml_declaration=True)
