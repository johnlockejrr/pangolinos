#!/usr/bin/env python3
"""
Visualize polygons from ALTO XML output to verify they follow character shapes.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import re

def parse_alto_points(points_str):
    """Parse ALTO POINTS attribute into list of (x, y) tuples."""
    points = []
    # Split by spaces and parse each "x,y" pair
    for point_str in points_str.strip().split():
        if ',' in point_str:
            x, y = point_str.split(',', 1)
            try:
                points.append((int(float(x)), int(float(y))))
            except ValueError:
                continue
    return points

def visualize_polygons(xml_file, output_image):
    """Read ALTO XML and draw polygons on an image."""
    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get page dimensions
    page = root.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Page')
    if page is None:
        print("No Page element found")
        return
    
    width = int(page.get('WIDTH', 1000))
    height = int(page.get('HEIGHT', 1000))
    
    print(f"Page dimensions: {width} x {height}")
    
    # Create image with white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Find all TextLines with polygons
    textlines = root.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}TextLine')
    print(f"Found {len(textlines)} text lines")
    
    for i, line in enumerate(textlines):
        # Get text content
        string_elem = line.find('.//{http://www.loc.gov/standards/alto/ns-v4#}String')
        text = string_elem.get('CONTENT', '') if string_elem is not None else f'Line {i}'
        
        # Get polygon points
        polygon = line.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Polygon')
        if polygon is not None:
            points_str = polygon.get('POINTS', '')
            if points_str:
                points = parse_alto_points(points_str)
                print(f"Line '{text}': {len(points)} polygon points")
                
                if len(points) > 2:
                    # Draw polygon outline
                    draw.polygon(points, outline='red', width=2)
                    
                    # Draw points
                    for point in points:
                        draw.ellipse([point[0]-2, point[1]-2, point[0]+2, point[1]+2], 
                                   fill='blue', outline='blue')
                    
                    # Draw text label
                    if points:
                        # Find center of polygon
                        center_x = sum(p[0] for p in points) // len(points)
                        center_y = sum(p[1] for p in points) // len(points)
                        draw.text((center_x, center_y), text, fill='black')
                else:
                    print(f"  Not enough points for polygon: {points}")
            else:
                print(f"Line '{text}': No polygon points found")
        else:
            print(f"Line '{text}': No polygon element found")
    
    # Save image
    img.save(output_image)
    print(f"Visualization saved to {output_image}")

def main():
    parser = argparse.ArgumentParser(description='Visualize polygons from ALTO XML')
    parser.add_argument('xml_file', help='ALTO XML file to visualize')
    parser.add_argument('-o', '--output', default='polygon_visualization.png', 
                       help='Output image file')
    
    args = parser.parse_args()
    
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: {xml_file} not found")
        return 1
    
    visualize_polygons(xml_path, args.output)
    return 0

if __name__ == '__main__':
    exit(main())
