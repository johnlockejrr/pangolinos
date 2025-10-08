#!/usr/bin/env python3
"""
pangolinos.fine_contour_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine contour extraction using image processing techniques similar to Aletheia's
"To Fine Contour" functionality. This module adapts the to_fine_contour.py method
to work with pangolinos' data structures and coordinate systems.
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    id: Optional[str] = None


def ensure_odd(value: int) -> int:
    """Ensure value is odd and >= 1."""
    if value < 1:
        value = 1
    return value if value % 2 == 1 else value + 1


def to_bw_and_save(color_image: np.ndarray, out_path: str) -> np.ndarray:
    """Convert color image to binary (black/white) and save."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Otsu threshold, invert so text is black=0, background white=255
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = 255 - bin_inv  # text=0, background=255
    if not cv2.imwrite(out_path, bw):
        raise RuntimeError(f"Failed to write BW image: {out_path}")
    return bw


def find_external_contour(mask_255: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest external contour in a binary mask."""
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick the largest by area
    contour = max(contours, key=cv2.contourArea)
    return contour


def smear_and_select(
    roi_text_mask_255: np.ndarray,
    smear_start: int,
    smear_inc: int,
    include_all_pixels: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply iterative smearing to unify text components."""
    # roi_text_mask_255: 0 where text pixels, 255 background
    # Convert to 255 for text for morphological ops
    text_foreground = (roi_text_mask_255 == 0).astype(np.uint8) * 255

    if include_all_pixels:
        # Include all inside bbox; unified is just a copy
        selected_mask = text_foreground.copy()
        return selected_mask, selected_mask, 1

    kernel_size = ensure_odd(int(smear_start))
    inc = max(1, int(smear_inc))

    # Iteratively smear (dilate) until we form a single external blob
    max_iters = 100
    current = text_foreground
    selected_mask = np.zeros_like(text_foreground)
    unified_mask = np.zeros_like(text_foreground)
    kernel_used = kernel_size

    for _ in range(max_iters):
        # Prefer horizontal smearing to connect words without inflating vertically
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 3))
        dilated = cv2.dilate(text_foreground, kernel, iterations=1)

        # Count external contours; stop when 0 or 1
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 1:
            # Identify the unified region mask from the largest contour
            if len(contours) == 1:
                unified_mask = np.zeros_like(dilated)
                cv2.drawContours(unified_mask, contours, -1, color=255, thickness=cv2.FILLED)
            else:
                unified_mask = dilated
            # Select original components whose pixels lie inside the unified region
            selected_mask = cv2.bitwise_and(text_foreground, unified_mask)
            kernel_used = kernel_size
            break

        kernel_size = ensure_odd(kernel_size + inc)
        current = dilated

    if np.count_nonzero(selected_mask) == 0:
        # Fallback to all pixels to avoid empty selections
        selected_mask = text_foreground
        unified_mask = text_foreground
        kernel_used = 1

    return selected_mask, unified_mask, kernel_used


def contour_from_mask(
    foreground_mask_255: np.ndarray,
    padding: int,
) -> Optional[np.ndarray]:
    """Extract a single external contour from a (possibly smeared) mask."""
    mask = foreground_mask_255.copy()
    if padding > 0:
        k = ensure_odd(padding * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    contour = find_external_contour(mask)
    return contour


def process_bbox(
    bw_image_255: np.ndarray,
    bbox: BoundingBox,
    smear_start: int,
    smear_inc: int,
    include_all_pixels: bool,
    padding: int,
    max_merge_gap: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Process a single bounding box to extract fine contour."""
    h_img, w_img = bw_image_255.shape[:2]

    x0 = max(0, bbox.x)
    y0 = max(0, bbox.y)
    x1 = min(w_img, bbox.x + bbox.w)
    y1 = min(h_img, bbox.y + bbox.h)
    if x0 >= x1 or y0 >= y1:
        return None

    roi = bw_image_255[y0:y1, x0:x1]

    # Smear to unify line and also capture original selected pixels
    selected, unified, kernel_used = smear_and_select(
        roi_text_mask_255=roi,
        smear_start=smear_start,
        smear_inc=smear_inc,
        include_all_pixels=include_all_pixels,
    )

    # Connect words into a single component using morphological closing
    kx = max(ensure_odd(kernel_used), 3)
    ky = 3
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # If still multiple components, increase horizontal closing width progressively
    if max_merge_gap is not None and max_merge_gap > 0:
        max_extra = ensure_odd(int(max_merge_gap))
    else:
        max_extra = max(ensure_odd(bbox.w // 6), kx)  # cap additional width sensibly
    extra = 0
    for _ in range(10):
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 1:
            break
        extra = min(max_extra, extra + max(5, kx // 4))
        kxx = ensure_odd(kx + extra)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kxx, ky))
        closed = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Extract fine outer contour from the closed mask
    contour_local = contour_from_mask(closed, padding=padding)
    if contour_local is None:
        return None

    # Shift contour to global coordinates
    contour_global = contour_local + np.array([[x0, y0]], dtype=np.int32)
    return contour_global


def extract_fine_contours_from_alto(
    alto_xml_path: str,
    image_path: str,
    smear_start: int = 100,
    smear_inc: int = 100,
    include_all_pixels: bool = False,
    padding: int = 4,
    max_merge_gap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract fine contours from ALTO XML using image processing.
    
    Args:
        alto_xml_path: Path to ALTO XML file
        image_path: Path to corresponding image file
        smear_start: Start value for smearing kernel (pixels)
        smear_inc: Increment for smearing kernel (pixels)
        include_all_pixels: Include all black pixels inside bbox
        padding: Padding in pixels around final contour
        max_merge_gap: Max extra horizontal width to bridge during adaptive closing
        
    Returns:
        Dictionary with extracted contours and metadata
    """
    # Read ALTO XML to get bounding boxes
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()
    
    # Find TextLine elements
    textline_elems = root.findall(".//{*}TextLine")
    bboxes: List[BoundingBox] = []
    
    for tl in textline_elems:
        try:
            x = int(round(float(tl.attrib.get("HPOS", "0"))))
            y = int(round(float(tl.attrib.get("VPOS", "0"))))
            w = int(round(float(tl.attrib.get("WIDTH", "0"))))
            h = int(round(float(tl.attrib.get("HEIGHT", "0"))))
        except ValueError:
            continue

        if w <= 0 or h <= 0:
            continue

        tl_id = tl.attrib.get("ID")
        bboxes.append(BoundingBox(x=x, y=y, w=w, h=h, id=tl_id))
    
    # Load and process image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    
    # Create temporary BW image (Path.with_suffix requires a real suffix like '.png')
    img_path = Path(image_path)
    bw_temp_path = str(img_path.with_name(f"{img_path.stem}_bw_temp.png"))
    bw_image = to_bw_and_save(image, bw_temp_path)
    
    # Process each bounding box
    results: List[Dict[str, Any]] = []
    id_to_polygon: Dict[Optional[str], List[List[int]]] = {}
    
    for bbox in bboxes:
        contour = process_bbox(
            bw_image_255=bw_image,
            bbox=bbox,
            smear_start=smear_start,
            smear_inc=smear_inc,
            include_all_pixels=include_all_pixels,
            padding=padding,
            max_merge_gap=max_merge_gap,
        )
        
        poly_points: List[List[int]] = []
        if contour is not None:
            # Ensure integer points
            contour = contour.astype(np.int32)
            for pt in contour.reshape(-1, 2):
                poly_points.append([int(pt[0]), int(pt[1])])
        
        result_entry = {
            "id": bbox.id,
            "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
            "polygon": poly_points,
        }
        results.append(result_entry)
        
        if bbox.id and poly_points:
            id_to_polygon[bbox.id] = poly_points
    
    # Clean up temporary BW image
    try:
        Path(bw_temp_path).unlink()
    except OSError:
        pass
    
    return {
        "image": image_path,
        "parameters": {
            "smear_start": smear_start,
            "smear_inc": smear_inc,
            "include_all_pixels": include_all_pixels,
            "padding": padding,
        },
        "lines": results,
        "id_to_polygon": id_to_polygon,
    }


def write_alto_with_polygons(
    alto_xml_path: str,
    out_alto_path: str,
    id_to_polygon: Dict[Optional[str], List[List[int]]],
) -> str:
    """Write ALTO XML with polygon coordinates replaced."""
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()

    # Extract namespace
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0][1:]
    else:
        ns = ""

    # Ensure default namespace is preserved
    if ns:
        ET.register_namespace('', ns)

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    # Replace or insert Shape/Polygon for each TextLine with matching ID
    for tl in root.findall(f".//{{*}}TextLine"):
        tl_id = tl.attrib.get("ID")
        poly = id_to_polygon.get(tl_id)
        if not poly:
            continue

        # Format POINTS attribute
        points_str = " ".join(f"{x},{y}" for x, y in poly)

        shape = tl.find(f"{q('Shape')}")
        if shape is None:
            shape = ET.SubElement(tl, q('Shape'))

        polygon = shape.find(f"{q('Polygon')}")
        if polygon is None:
            polygon = ET.SubElement(shape, q('Polygon'))
        polygon.set("POINTS", points_str)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_alto_path)), exist_ok=True)
    tree.write(out_alto_path, encoding="utf-8", xml_declaration=True)
    return os.path.abspath(out_alto_path)
