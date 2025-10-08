#
# Copyright 2025 johnlockejrr / Benjamin Kiessling
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
pangolinos.rasterize
~~~~~~~~~~~~~~~~~~~
"""
import re
import copy
import logging
import pypdfium2 as pdfium

from PIL import Image
from lxml import etree
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from os import PathLike


@staticmethod
def _parse_alto_pointstype(coords: str) -> list[tuple[float, float]]:
    """
    ALTO's PointsType is underspecified so a variety of serializations are valid:

        x0, y0 x1, y1 ...
        x0 y0 x1 y1 ...
        (x0, y0) (x1, y1) ...
        (x0 y0) (x1 y1) ...

    Returns:
        A list of tuples [(x0, y0), (x1, y1), ...]
    """
    float_re = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
    points = [float(point.group()) for point in float_re.finditer(coords)]
    if len(points) % 2:
        raise ValueError(f'Odd number of points in points sequence: {points}')
    return list(zip(points[::2], points[1::2]))


def rasterize_document(doc: Union[str, 'PathLike'],
                       output_base_path: Union[str, 'PathLike'],
                       writing_surface: Optional[Union[str, 'PathLike']] = None,
                       dpi: int = 300,
                       use_polygons: bool = False,
                       use_fine_contours: bool = False,
                       smear_start: int = 100,
                       smear_inc: int = 100,
                       include_all_pixels: bool = False,
                       padding: int = 4):
    """
    Takes an ALTO XML file, rasterizes the associated PDF document with the
    given resolution and rewrites the ALTO, translating the physical dimension
    to pixel positions.

    The output image and XML files will be at `output_base_path/doc`.

    Args:
        doc: Input ALTO file
        output_base_path: Directory to write output image file and rewritten
                          ALTO into.
        writing_surface: Path to image file used as a background writing
                         surface on which the rasterized text is pasted on.
                         The image will be resized to the selected PDF
                         resolution.
        dpi: DPI to render the PDF
        use_polygons: Process polygon coordinates instead of rectangular bounding boxes
        use_fine_contours: Process fine contours using image processing (excludes use_polygons)
        smear_start: Start value for smearing kernel (pixels). Only used with use_fine_contours
        smear_inc: Increment for smearing kernel (pixels). Only used with use_fine_contours
        include_all_pixels: Include all black pixels inside bbox. Only used with use_fine_contours
        padding: Padding in pixels around final contour. Only used with use_fine_contours

    """
    # Validation: can't use both polygon methods
    if use_polygons and use_fine_contours:
        raise ValueError("Cannot use both use_polygons and use_fine_contours. Choose one.")
    
    output_base_path = Path(output_base_path)
    doc = Path(doc)

    if writing_surface:
        writing_surface = Image.open(writing_surface).convert('RGBA')

    coord_scale = dpi / 25.4
    _dpi_point = 1 / 72

    tree = etree.parse(doc)
    tree.find('.//{*}MeasurementUnit').text = 'pixel'
    fileName = tree.find('.//{*}fileName')
    pdf_file = doc.parent / fileName.text
    # rasterize and save as png
    pdf_page = pdfium.PdfDocument(pdf_file).get_page(0)
    transparency = 0 if writing_surface else 255
    im = pdf_page.render(scale=dpi*_dpi_point, fill_color=(255, 255, 255, transparency)).to_pil()
    if writing_surface:
        writing_surface = writing_surface.resize(im.size)
        writing_surface.alpha_composite(im)
        im = writing_surface

    fileName.text = doc.with_suffix('.png').name

    # Save primary raster (with background if provided)
    out_image_path = output_base_path / fileName.text
    im.save(out_image_path, format='png', optimize=True)

    # Process fine contours if requested
    if use_fine_contours:
        logger.info("Processing fine contours using image processing...")
        try:
            from pangolinos.fine_contour_extractor import extract_fine_contours_from_alto

            # If a background writing_surface is used, generate a second raster without background
            created_nobg = False
            nobg_im_path = out_image_path
            if writing_surface is not None:
                nobg_im_path = output_base_path / f"{doc.stem}_nobg.png"
                pdf_page_clean = pdfium.PdfDocument(pdf_file).get_page(0)
                clean_im = pdf_page_clean.render(scale=dpi*_dpi_point, fill_color=(255, 255, 255, 255)).to_pil()
                clean_im.save(nobg_im_path, format='png', optimize=True)
                created_nobg = True

            # Build a temporary ALTO with pixel coordinates for accurate ROI cropping
            tmp_tree = copy.deepcopy(tree)
            tmp_root = tmp_tree.getroot()
            mu_node = tmp_tree.find('.//{*}MeasurementUnit')
            if mu_node is not None:
                mu_node.text = 'pixel'
            tmp_page = tmp_tree.find('.//{*}Page')
            tmp_printspace = tmp_tree.find('.//{*}PrintSpace')
            if tmp_page is not None:
                tmp_page.set('WIDTH', str(im.width))
                tmp_page.set('HEIGHT', str(im.height))
            if tmp_printspace is not None:
                tmp_printspace.set('WIDTH', str(im.width))
                tmp_printspace.set('HEIGHT', str(im.height))

            for tline in tmp_tree.findall('.//{*}TextLine'):
                hpos = int(float(tline.get('HPOS')) * coord_scale)
                vpos = int(float(tline.get('VPOS')) * coord_scale)
                width_px = int(float(tline.get('WIDTH')) * coord_scale)
                height_px = int(float(tline.get('HEIGHT')) * coord_scale)
                tline.set('HPOS', str(hpos))
                tline.set('VPOS', str(vpos))
                tline.set('WIDTH', str(width_px))
                tline.set('HEIGHT', str(height_px))
                baseline_points = _parse_alto_pointstype(tline.get('BASELINE'))
                if len(baseline_points) == 2:
                    (bl_x0, bl_y0), (bl_x1, bl_y1) = baseline_points
                    tline.set('BASELINE', f'{int(bl_x0 * coord_scale)},{int(bl_y0 * coord_scale)} {int(bl_x1 * coord_scale)},{int(bl_y1 * coord_scale)}')

            tmp_alto_path = output_base_path / f"{doc.stem}.pixels.tmp.xml"
            tmp_tree.write(tmp_alto_path, encoding='utf-8')

            # Extract fine contours from the clean image (or the main image if no background)
            contours_result = extract_fine_contours_from_alto(
                alto_xml_path=str(tmp_alto_path),
                image_path=str(nobg_im_path),
                smear_start=smear_start,
                smear_inc=smear_inc,
                include_all_pixels=include_all_pixels,
                padding=padding
            )

            # Clean up temporary no-bg image if we created it
            if created_nobg:
                try:
                    nobg_im_path.unlink()
                except OSError:
                    pass

            # Remove temporary ALTO file
            try:
                tmp_alto_path.unlink()
            except OSError:
                pass

            logger.info(f"Extracted fine contours for {len(contours_result['lines'])} lines")

        except Exception as e:
            logger.error(f"Failed to extract fine contours: {e}")
            logger.warning("Falling back to bounding box processing")
            use_fine_contours = False

    # rewrite coordinates
    page = tree.find('.//{*}Page')
    printspace = page.find('./{*}PrintSpace')
    page.set('WIDTH', str(im.width))
    page.set('HEIGHT', str(im.height))
    printspace.set('WIDTH', str(im.width))
    printspace.set('HEIGHT', str(im.height))

    for line in tree.findall('.//{*}TextLine'):
        hpos = int(float(line.get('HPOS')) * coord_scale)
        vpos = int(float(line.get('VPOS')) * coord_scale)
        width = int(float(line.get('WIDTH')) * coord_scale)
        height = int(float(line.get('HEIGHT')) * coord_scale)
        line.set('HPOS', str(hpos))
        line.set('VPOS', str(vpos))
        line.set('WIDTH', str(width))
        line.set('HEIGHT', str(height))
        baseline_points = _parse_alto_pointstype(line.get('BASELINE'))
        if len(baseline_points) == 2:
            (bl_x0, bl_y0), (bl_x1, bl_y1) = baseline_points
            line.set('BASELINE', f'{int(bl_x0 * coord_scale)},{int(bl_y0 * coord_scale)} {int(bl_x1 * coord_scale)},{int(bl_y1 * coord_scale)}')
        else:
            logger.warning(f"Unexpected baseline format: {baseline_points}")
        
        # Handle polygon coordinates
        pol = line.find('.//{*}Polygon')
        if use_fine_contours and 'contours_result' in locals():
            # Ensure Shape/Polygon exists
            shape = line.find('./{*}Shape')
            if shape is None:
                shape = etree.SubElement(line, 'Shape')
            pol = shape.find('./{*}Polygon')
            if pol is None:
                pol = etree.SubElement(shape, 'Polygon')

        if pol is not None:
            if use_fine_contours and 'contours_result' in locals():
                # Use fine contours extracted from image processing
                line_id = line.get('ID')
                fine_contour = None
                
                # Find matching contour by ID
                id_to_poly = contours_result.get('id_to_polygon') or {}
                fine_contour = id_to_poly.get(line_id)
                
                if fine_contour and len(fine_contour) > 0:
                    # Scale fine contour points to pixel coordinates
                    scaled_points = []
                    for px, py in fine_contour:
                        scaled_x = int(px)
                        scaled_y = int(py)
                        scaled_points.append(f'{scaled_x},{scaled_y}')
                    final_points = ' '.join(scaled_points)
                    pol.set('POINTS', final_points)
                    logger.info(f"Applied fine contour with {len(fine_contour)} points for line {line_id}")
                else:
                    # Fallback to rectangle if no fine contour found
                    pol.set('POINTS', f'{hpos},{vpos} {hpos+width},{vpos} {hpos+width},{vpos+height} {hpos},{vpos+height}')
                    logger.warning(f"No fine contour found for line {line_id}, using rectangle")
            else:
                # Handle existing polygon coordinates (use_polygons mode)
                current_points = pol.get('POINTS', '')
                if current_points:
                    try:
                        # Parse existing points
                        points = _parse_alto_pointstype(current_points)
                        logger.info(f"Successfully parsed {len(points)} points from: {current_points[:100]}...")
                        
                        # Always scale all polygon points, regardless of count
                        # This preserves detailed FreeType-extracted polygons
                        logger.info(f"Scaling {len(points)} polygon points with coord_scale={coord_scale:.3f}")
                        logger.info(f"Original points sample: {points[:5]}")
                        scaled_points = []
                        for px, py in points:
                            scaled_x = int(px * coord_scale)
                            scaled_y = int(py * coord_scale)
                            scaled_points.append(f'{scaled_x},{scaled_y}')
                        logger.info(f"Scaled to {len(scaled_points)} points: {scaled_points[:5]}...")
                        final_points = ' '.join(scaled_points)
                        logger.info(f"Setting POINTS to: {final_points[:100]}...")
                        pol.set('POINTS', final_points)
                    except (ValueError, TypeError) as e:
                        # If parsing fails, fall back to rectangle method
                        logger.error(f"Failed to parse polygon points: {e}")
                        logger.error(f"Points string was: {current_points[:100]}...")
                        pol.set('POINTS', f'{hpos},{vpos} {hpos+width},{vpos} {hpos+width},{vpos+height} {hpos},{vpos+height}')
                else:
                    # No existing points, create rectangle
                    pol.set('POINTS', f'{hpos},{vpos} {hpos+width},{vpos} {hpos+width},{vpos+height} {hpos},{vpos+height}')
    tree.write(output_base_path / doc.name, encoding='utf-8')
