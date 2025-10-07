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
pangolinos.polygon_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~

FreeType-based polygon extraction for exact glyph outlines.
"""
import logging
import os
import subprocess
import freetype
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional
from gi.repository import Pango

logger = logging.getLogger(__name__)


def extract_line_polygons(line, 
                         layout,
                         lleft: float, 
                         bl: float, 
                         left_margin: float, 
                         top_margin: float, 
                         _mm_point: float,
                         font_face: freetype.Face) -> List[Tuple[float, float]]:
    """
    Extract exact polygon coordinates from a Pango line using FreeType glyph outlines.
    Returns a list of (x, y) tuples in points.
    """
    try:
        # Get the actual font size from the layout
        layout_font_desc = layout.get_font_description()
        if layout_font_desc:
            font_size_pt = int(layout_font_desc.get_size() / Pango.SCALE)
        else:
            font_size_pt = 12  # Default fallback
        
        # If font size is still 0, use the font face's actual size
        if font_size_pt <= 0:
            font_size_pt = int(font_face.size.height / 64.0)  # Convert from 26.6 fixed point
        
        # Scale factor: font units → points
        units_per_em = font_face.units_per_EM
        scale_factor = font_size_pt / units_per_em

        glyph_polys = []
        
        # Use PangoCairo to get individual glyph outlines, then create line envelope
        import cairo
        from gi.repository import PangoCairo
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon
        
        # Create a temporary Cairo surface and context
        surface = cairo.ImageSurface(cairo.Format.ARGB32, 1, 1)
        context = cairo.Context(surface)
        
        # Get the line path from PangoCairo (gives individual glyph paths)
        context.save()
        context.move_to(lleft, bl)
        PangoCairo.layout_line_path(context, line)
        path = context.copy_path_flat()
        context.restore()
        
        # Split into individual glyph polygons - fix to get ALL characters
        polys = []
        current = []
        path_list = list(path)  # Convert to list so we can iterate multiple times
        
        for i, (op, payload) in enumerate(path_list):
            if op == cairo.PATH_MOVE_TO:
                # Start a new polygon - close the previous one first
                if len(current) >= 3:
                    poly = Polygon(current)
                    if not poly.is_empty:
                        polys.append(poly)
                current = [payload]  # payload is (x, y)
            elif op == cairo.PATH_LINE_TO:
                current.append(payload)
            elif op == cairo.PATH_CLOSE_PATH:
                if len(current) >= 3:
                    poly = Polygon(current)
                    if not poly.is_empty:
                        polys.append(poly)
                current = []  # Reset for next glyph
        
        # Add the last polygon if it exists
        if len(current) >= 3:
            poly = Polygon(current)
            if not poly.is_empty:
                polys.append(poly)
        
        logger.info(f"Extracted {len(polys)} individual glyph polygons from Cairo path")
        
        if not polys:
            logger.warning("No polygons extracted from Cairo path")
            return []
        
        # Clean each glyph polygon
        cleaned = []
        for p in polys:
            try:
                q = p.buffer(0)
                if q.is_empty:
                    continue
                if q.geom_type == 'Polygon':
                    cleaned.append(q)
                elif q.geom_type == 'MultiPolygon':
                    cleaned.extend(list(q.geoms))
                else:
                    # GeometryCollection: take polygonal parts
                    cleaned.extend([g for g in getattr(q, 'geoms', []) if g.geom_type == 'Polygon'])
            except Exception as e:
                logger.warning(f"Failed to clean polygon: {e}")
                continue
        
        if not cleaned:
            logger.warning("No valid polygons after cleaning")
            return []
        
        # Strategy: Preserve glyph shapes but bridge gaps between characters
        # Use minimal buffering to connect nearby glyphs while preserving contours
        
        # First, try direct union of cleaned glyphs (no buffering)
        try:
            unioned = unary_union(cleaned)
            logger.info(f"Direct union result type: {unioned.geom_type}")
        except Exception as e:
            logger.warning(f"Direct union failed: {e}")
            from shapely.geometry import MultiPolygon
            unioned = MultiPolygon(cleaned)

        def handle_multipolygon(geom):
            """Convert MultiPolygon to single Polygon by creating envelope around all parts."""
            if hasattr(geom, 'exterior'):
                return geom  # Already a Polygon
            elif hasattr(geom, 'geoms'):
                parts = list(geom.geoms)
                if len(parts) == 1:
                    return parts[0]
                elif len(parts) > 1:
                    logger.info(f"MultiPolygon with {len(parts)} parts - creating envelope to preserve all words")
                    
                    # Strategy 1: Try a small buffer to connect all parts
                    try:
                        # Use a slightly larger buffer to connect separated word groups
                        connect_buffer_mm = 0.8  # Larger buffer to connect words
                        connect_pt = connect_buffer_mm * _mm_point
                        connected = geom.buffer(connect_pt, cap_style=1, join_style=1, resolution=8)
                        
                        if hasattr(connected, 'exterior'):
                            # Successfully connected - now shrink back
                            shrunk = connected.buffer(-connect_pt * 0.6, cap_style=1, join_style=1, resolution=8)
                            if hasattr(shrunk, 'exterior') and not shrunk.is_empty:
                                logger.info(f"Successfully connected {len(parts)} parts with buffering")
                                return shrunk
                    except Exception as e:
                        logger.warning(f"Buffer connection failed: {e}")
                    
                    # Strategy 2: Create convex hull as envelope (preserves all words)
                    try:
                        from shapely.geometry import MultiPoint
                        all_points = []
                        for part in parts:
                            if hasattr(part, 'exterior'):
                                all_points.extend(list(part.exterior.coords[:-1]))
                        
                        if len(all_points) >= 3:
                            envelope = MultiPoint(all_points).convex_hull
                            if hasattr(envelope, 'exterior'):
                                logger.info(f"Created convex hull envelope for {len(parts)} separated word groups")
                                return envelope
                    except Exception as e:
                        logger.warning(f"Convex hull envelope failed: {e}")
                    
                    # Strategy 3: Create bounding box envelope (last resort but preserves all words)
                    try:
                        from shapely.geometry import box
                        min_x = min_y = float('inf')
                        max_x = max_y = float('-inf')
                        
                        for part in parts:
                            bounds = part.bounds
                            min_x = min(min_x, bounds[0])
                            min_y = min(min_y, bounds[1])
                            max_x = max(max_x, bounds[2])
                            max_y = max(max_y, bounds[3])
                        
                        # Add small padding to bounding box
                        padding = 0.2 * _mm_point
                        envelope = box(min_x - padding, min_y - padding, 
                                     max_x + padding, max_y + padding)
                        logger.info(f"Created bounding box envelope for {len(parts)} separated parts")
                        return envelope
                        
                    except Exception as e:
                        logger.warning(f"Bounding box envelope failed: {e}")
                    
                    # Last resort: return largest component (but warn about data loss)
                    largest = max(parts, key=lambda p: p.area if hasattr(p, 'area') else 0)
                    logger.warning(f"ALL envelope strategies failed - using largest component only (DATA LOSS!)")
                    return largest
            return None

        # If we get a single polygon, we're done - glyphs were already touching
        if hasattr(unioned, 'exterior'):
            final_poly = unioned
            logger.info("Glyphs formed single polygon without buffering")
        
        # If we have multiple parts, we need to bridge gaps carefully
        elif hasattr(unioned, 'geoms'):
            parts = list(unioned.geoms)
            logger.info(f"Got {len(parts)} separate glyph parts, attempting careful bridging")
            
            # Try progressive bridging with minimal buffer sizes
            bridge_attempts = [0.1, 0.3, 0.5, 1.0, 1.5]  # mm
            final_poly = None
            
            for bridge_mm in bridge_attempts:
                bridge_pt = bridge_mm * _mm_point
                try:
                    # Very small buffer to bridge tiny gaps
                    bridged = unioned.buffer(bridge_pt, cap_style=1, join_style=1, resolution=16)
                    
                    # Check if this created a single connected component
                    if hasattr(bridged, 'exterior'):
                        # Success! Now shrink back to preserve original shape
                        try:
                            final_poly = bridged.buffer(-bridge_pt * 0.7, cap_style=1, join_style=1, resolution=16)
                            # Ensure result is still a single polygon
                            final_poly = handle_multipolygon(final_poly)
                            if final_poly:
                                logger.info(f"Successfully bridged with {bridge_mm}mm buffer")
                                break
                        except Exception as e:
                            logger.warning(f"Failed to shrink bridged polygon: {e}")
                            final_poly = handle_multipolygon(bridged)
                            if final_poly:
                                logger.info(f"Using bridged polygon without shrinking")
                                break
                    elif hasattr(bridged, 'geoms'):
                        new_parts = list(bridged.geoms)
                        if len(new_parts) < len(parts):
                            # Reduced number of parts, keep trying with this result
                            unioned = bridged
                            parts = new_parts
                            logger.info(f"Reduced to {len(new_parts)} parts with {bridge_mm}mm buffer")
                        elif len(new_parts) == 1:
                            # Got single part but it's wrapped in MultiPolygon
                            final_poly = handle_multipolygon(bridged)
                            if final_poly:
                                logger.info(f"Successfully unified to single part")
                                break
                        
                except Exception as e:
                    logger.warning(f"Bridging attempt {bridge_mm}mm failed: {e}")
                    continue
            
            # If bridging failed, take the largest component
            if not final_poly:
                final_poly = handle_multipolygon(unioned)
                if final_poly:
                    logger.info("Bridging failed, using envelope of all components")
                    # Add very small buffer to smooth any rough edges
                    try:
                        smoothed = final_poly.buffer(0.05 * _mm_point, cap_style=1, join_style=1)
                        final_poly = handle_multipolygon(smoothed) or final_poly
                    except Exception:
                        pass
        else:
            logger.warning("Unexpected union result")
            return []

        # Final check that we have a valid polygon
        if not final_poly or not hasattr(final_poly, 'exterior'):
            logger.warning("Failed to create valid polygon")
            return []

        # Minimal smoothing while preserving character details
        # Only simplify if we have many points (>100)
        try:
            if len(final_poly.exterior.coords) > 100:
                # Very conservative simplification - just remove collinear points
                simplified = final_poly.simplify(0.02 * _mm_point, preserve_topology=True)
                final_poly = handle_multipolygon(simplified) or final_poly
        except Exception as e:
            logger.warning(f"Simplification failed: {e}")
            pass

        # Convert to mm and return exterior ring
        coords_mm = []
        for x_pt, y_pt in final_poly.exterior.coords:
            x_mm = round(x_pt / _mm_point, 2)
            y_mm = round(y_pt / _mm_point, 2)
            coords_mm.append((x_mm, y_mm))

        logger.info(f"Extracted line envelope with {len(coords_mm)} points from {len(cleaned)} glyphs; final type={final_poly.geom_type}")
        return coords_mm

    except Exception as e:
        logger.error(f"Error extracting polygon from line: {e}")
        return []


def glyph_outline(face: freetype.Face, 
                 glyph_index: int, 
                 x_offset: float = 0, 
                 y_offset: float = 0) -> List[Polygon]:
    """
    Extract a glyph outline as a polygon (flattened).
    
    Args:
        face: FreeType face object
        glyph_index: Index of the glyph to extract
        x_offset: X offset for positioning (in font units)
        y_offset: Y offset for positioning (in font units)
        
    Returns:
        List of Polygon objects representing the glyph contours in font units
    """
    try:
        face.load_glyph(glyph_index, freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
        outline = face.glyph.outline
        
        points = outline.points
        contours = outline.contours
        
        start = 0
        polys = []
        for end in contours:
            contour_points = points[start:end+1]
            # Return raw font units, scaling will be done later
            poly = [(x_offset + p[0], y_offset - p[1]) for p in contour_points]
            if len(poly) >= 3:
                polys.append(Polygon(poly))
            start = end + 1
        return polys
    except Exception as e:
        logger.error(f"Error extracting glyph outline: {e}")
        return []


def _fc_match_file(font_pattern: str) -> Optional[str]:
    """Resolve a font file path using fontconfig (fc-match)."""
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", font_pattern],
            check=False,
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        if path and os.path.exists(path):
            return path
        return None
    except Exception:
        return None


def _font_matches_family(font_path: str, family_name: str) -> bool:
    """Check if a font file matches the expected family name."""
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}", font_path],
            check=False,
            capture_output=True,
            text=True,
        )
        font_family = result.stdout.strip()
        return family_name.lower() in font_family.lower()
    except Exception:
        return False


def get_font_face(font_string: str, font_size: int) -> Optional[freetype.Face]:
    """
    Get a FreeType face for the given font string and size.
    
    Args:
        font_string: Font description string (e.g., "Serif Normal 10")
        font_size: Font size in points
        
    Returns:
        FreeType face object or None if font cannot be loaded
    """
    try:
        # Extract font family name from font string (remove size and style)
        # e.g., "Shlomo Stam 24" -> "Shlomo Stam"
        font_family = font_string.split()[0] if font_string.split() else font_string
        
        # First try to resolve via fontconfig for the exact font string
        font_path = _fc_match_file(font_string)
        logger.info(f"First attempt '{font_string}': {font_path}")
        
        # Check if the returned font actually matches our request
        if font_path and not _font_matches_family(font_path, font_family):
            logger.info(f"Font doesn't match family '{font_family}', trying family name only")
            font_path = None
            
        if not font_path:
            # Try with just the family name
            font_path = _fc_match_file(font_family)
            logger.info(f"Second attempt '{font_family}': {font_path}")
        if not font_path:
            # Try to parse with Pango and resolve family + style via fontconfig
            try:
                desc = Pango.font_description_from_string(font_string)
                family = desc.get_family() or ""
                logger.info(f"Pango parsed family: '{family}' from '{font_string}'")
                style_bits = []
                weight = desc.get_weight()
                if weight and int(weight) >= int(Pango.Weight.BOLD):
                    style_bits.append("Bold")
                style = desc.get_style()
                if style and int(style) != int(Pango.Style.NORMAL):
                    style_bits.append("Italic")
                pattern = family
                if style_bits:
                    pattern = f"{family}:style={' '.join(style_bits)}"
                logger.info(f"Trying fontconfig pattern: '{pattern}'")
                if pattern.strip():
                    font_path = _fc_match_file(pattern)
                    logger.info(f"Fontconfig result: {font_path}")
            except Exception as e:
                logger.warning(f"Pango parsing failed: {e}")
                font_path = None

        if font_path:
            try:
                face = freetype.Face(font_path)
                # Try to set the font size, but don't worry if it doesn't work
                # We'll use the actual loaded size and adjust the scale factor
                try:
                    face.set_pixel_sizes(0, int(font_size * 1.33))  # Convert points to pixels (72 DPI)
                except:
                    pass  # Ignore if set_pixel_sizes fails
                
                logger.info(f"Successfully loaded font via fontconfig: {font_path}")
                logger.info(f"Requested font size: {font_size}pt")
                actual_size = face.size.height / 64.0
                logger.info(f"Actual font size after loading: {actual_size}pt")
                
                # Store the actual size in the face object for later use
                face._requested_size = font_size
                face._actual_size = actual_size
                
                return face
            except Exception as e:
                logger.warning(f"Failed to load resolved font {font_path}: {e}")

        # As a final fallback, let fontconfig choose a default sans-serif
        fallback = _fc_match_file("sans-serif")
        if fallback:
            try:
                face = freetype.Face(fallback)
                face.set_char_size(font_size * 64)
                logger.info(f"Using fallback font via fontconfig: {fallback}")
                return face
            except Exception as e:
                logger.warning(f"Failed to load fallback font {fallback}: {e}")

        logger.error("Could not resolve a font via fontconfig")
        return None
    except Exception as e:
        logger.error(f"Error loading font face: {e}")
        return None
