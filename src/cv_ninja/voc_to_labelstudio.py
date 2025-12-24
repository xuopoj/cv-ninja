#!/usr/bin/env python3
"""Convert Pascal VOC annotations to Label Studio JSON format."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML file and extract annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Use actual XML filename instead of reading from XML content
    filename = Path(xml_path).stem + '.jpg'
    size = root.find('size')
    width = 1516
    height = 1386
    # width = int(size.find('width').text)
    # height = int(size.find('height').text)

    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to Label Studio percentage format
        x_percent = (xmin / width) * 100
        y_percent = (ymin / height) * 100
        w_percent = ((xmax - xmin) / width) * 100
        h_percent = ((ymax - ymin) / height) * 100

        annotations.append({
            'from_name': 'label',
            'to_name': 'image',
            'type': 'rectanglelabels',
            'value': {
                'x': x_percent,
                'y': y_percent,
                'width': w_percent,
                'height': h_percent,
                'rectanglelabels': [name]
            },
            'original_width': width,
            'original_height': height
        })

    return filename, width, height, annotations


def convert_voc_to_labelstudio(voc_dir, output_path, prefix=''):
    """Convert all VOC XML files in directory to Label Studio format."""
    voc_path = Path(voc_dir)
    tasks = []

    for xml_file in sorted(voc_path.glob('*.xml')):
        filename, width, height, annotations = parse_voc_xml(xml_file)

        # Extract unique labels for column display
        labels = []
        if annotations:
            labels = list(set(ann['value']['rectanglelabels'][0] for ann in annotations))

        task = {
            'data': {
                'image': f'{prefix}{filename}',
                'label': ', '.join(labels) if labels else ''
            }
        }

        if annotations:
            task['annotations'] = [{
                'result': annotations
            }]

        tasks.append(task)

    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f'Converted {len(tasks)} tasks to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Pascal VOC to Label Studio format')
    parser.add_argument('input_dir', help='Directory containing Pascal VOC XML files')
    parser.add_argument('-o', '--output', default='labelstudio.json', help='Output JSON file')
    parser.add_argument('-p', '--prefix', default='', help='Prefix for image paths (e.g., /data/images/)')

    args = parser.parse_args()
    convert_voc_to_labelstudio(args.input_dir, args.output, args.prefix)
