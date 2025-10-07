import os
import re

def fix_baseline_coordinates(text):
    # Match BASELINE="..." and replace commas inside the quoted string with spaces
    def replacer(match):
        content = match.group(1)
        fixed = re.sub(r'(\d+),(\d+)', r'\1 \2', content)
        return f'BASELINE="{fixed}"'
    return re.sub(r'BASELINE="([^"]+)"', replacer, text)

def fix_polygon_points(text):
    # Match POINTS="..." and replace commas inside the quoted string with spaces
    def replacer(match):
        content = match.group(1)
        fixed = re.sub(r'(\d+),(\d+)', r'\1 \2', content)
        return f'POINTS="{fixed}"'
    return re.sub(r'POINTS="([^"]+)"', replacer, text)

def process_xml_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = fix_baseline_coordinates(content)
    new_content = fix_polygon_points(new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")
    else:
        print(f"No change: {filepath}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.xml'):
                process_xml_file(os.path.join(root, file))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fix_coords.py /path/to/xml_directory")
        sys.exit(1)

    target_dir = sys.argv[1]
    process_directory(target_dir)

