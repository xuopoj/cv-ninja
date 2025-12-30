"""Convert LabelMe annotations to Label Studio JSON format."""

import json
from pathlib import Path
from typing import List, Tuple

from cv_ninja.converters.base import BaseConverter
from cv_ninja.utils.exceptions import ConversionError, ValidationError


class LabelMeToLabelStudioConverter(BaseConverter):
    """Converter for LabelMe to Label Studio JSON format.

    Converts LabelMe JSON annotation files to Label Studio's JSON format
    with bounding box and polygon annotations as percentage coordinates.
    """

    def __init__(self, prefix: str = ''):
        """Initialize the converter.

        Args:
            prefix: URL prefix to prepend to image paths (e.g., '/data/images/')
        """
        self.prefix = prefix

    @property
    def supported_formats(self) -> Tuple[str, str]:
        """Return supported format conversion."""
        return ('labelme', 'labelstudio')

    def validate_input(self, input_path: str) -> bool:
        """Validate that input directory exists and contains JSON files.

        Args:
            input_path: Directory path containing LabelMe JSON files

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        path = Path(input_path)

        if not path.exists():
            raise ValidationError(f"Input path does not exist: {input_path}")

        if not path.is_dir():
            raise ValidationError(f"Input path is not a directory: {input_path}")

        json_files = list(path.glob('*.json'))
        if not json_files:
            raise ValidationError(f"No JSON files found in directory: {input_path}")

        return True

    def convert(self, input_dir: str, output_path: str, **kwargs) -> None:
        """Convert all LabelMe JSON files in directory to Label Studio format.

        Args:
            input_dir: Directory containing LabelMe JSON files
            output_path: Output JSON file path
            **kwargs: Additional options (unused)

        Raises:
            ConversionError: If conversion fails
        """
        try:
            self.validate_input(input_dir)
            labelme_path = Path(input_dir)
            tasks = []

            for json_file in sorted(labelme_path.glob('*.json')):
                try:
                    filename, annotations = self._parse_labelme_json(json_file)

                    # Extract unique labels for column display
                    labels = []
                    if annotations:
                        # Collect all labels from annotations
                        label_set = set()
                        for ann in annotations:
                            if ann['type'] == 'rectanglelabels':
                                label_set.update(ann['value']['rectanglelabels'])
                            elif ann['type'] == 'polygonlabels':
                                label_set.update(ann['value']['polygonlabels'])
                        labels = sorted(list(label_set))

                    task = {
                        'data': {
                            'image': f'{self.prefix}{filename}',
                            'label': ', '.join(labels) if labels else ''
                        }
                    }

                    if annotations:
                        task['annotations'] = [{
                            'result': annotations
                        }]

                    tasks.append(task)

                except Exception as e:
                    raise ConversionError(
                        f"Failed to convert {json_file.name}: {str(e)}"
                    ) from e

            # Write output JSON
            with open(output_path, 'w') as f:
                json.dump(tasks, f, indent=2)

        except ValidationError:
            raise
        except Exception as e:
            raise ConversionError(f"Conversion failed: {str(e)}") from e

    def _parse_labelme_json(self, json_path: Path) -> Tuple[str, List[dict]]:
        """Parse LabelMe JSON file and extract annotations.

        Args:
            json_path: Path to LabelMe JSON file

        Returns:
            Tuple of (filename, annotations)

        Raises:
            ConversionError: If parsing fails
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract image metadata
            filename = data.get('imagePath', json_path.stem + '.jpg')
            width = data.get('imageWidth')
            height = data.get('imageHeight')

            if not width or not height:
                raise ConversionError(
                    f"Missing imageWidth or imageHeight in {json_path.name}"
                )

            if width <= 0 or height <= 0:
                raise ConversionError(
                    f"Invalid dimensions ({width}x{height}) in {json_path.name}"
                )

            # Process shapes
            annotations = []
            shapes = data.get('shapes', [])

            for shape in shapes:
                label = shape.get('label')
                if not label:
                    continue  # Skip shapes without label

                shape_type = shape.get('shape_type', 'rectangle')
                points = shape.get('points', [])

                if not points:
                    continue  # Skip shapes without points

                # Convert based on shape type
                if shape_type == 'rectangle':
                    annotation = self._convert_rectangle(
                        label, points, width, height
                    )
                    if annotation:
                        annotations.append(annotation)

                elif shape_type == 'polygon':
                    annotation = self._convert_polygon(
                        label, points, width, height
                    )
                    if annotation:
                        annotations.append(annotation)

                # Add support for other shape types as needed
                # (circle, line, point, etc.)

            return filename, annotations

        except json.JSONDecodeError as e:
            raise ConversionError(
                f"Failed to parse JSON file {json_path.name}: {str(e)}"
            ) from e
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Unexpected error parsing {json_path.name}: {str(e)}"
            ) from e

    def _convert_rectangle(
        self, label: str, points: List[List[float]], width: int, height: int
    ) -> dict:
        """Convert rectangle annotation to Label Studio format.

        Args:
            label: Object label/class name
            points: List of [x, y] coordinates (2 points for rectangle)
            width: Image width
            height: Image height

        Returns:
            Label Studio rectangle annotation dict, or None if invalid
        """
        if len(points) != 2:
            return None

        try:
            # LabelMe rectangles: [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Ensure correct order (top-left to bottom-right)
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            # Convert to percentage coordinates
            x_percent = (xmin / width) * 100
            y_percent = (ymin / height) * 100
            w_percent = ((xmax - xmin) / width) * 100
            h_percent = ((ymax - ymin) / height) * 100

            return {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'x': x_percent,
                    'y': y_percent,
                    'width': w_percent,
                    'height': h_percent,
                    'rectanglelabels': [label]
                },
                'original_width': width,
                'original_height': height
            }

        except (ValueError, TypeError, IndexError):
            return None

    def _convert_polygon(
        self, label: str, points: List[List[float]], width: int, height: int
    ) -> dict:
        """Convert polygon annotation to Label Studio format.

        Args:
            label: Object label/class name
            points: List of [x, y] coordinates for polygon vertices
            width: Image width
            height: Image height

        Returns:
            Label Studio polygon annotation dict, or None if invalid
        """
        if len(points) < 3:
            return None  # Polygon needs at least 3 points

        try:
            # Convert points to percentage coordinates
            polygon_points = []
            for x, y in points:
                x_percent = (x / width) * 100
                y_percent = (y / height) * 100
                polygon_points.append([x_percent, y_percent])

            return {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'polygonlabels',
                'value': {
                    'points': polygon_points,
                    'polygonlabels': [label]
                },
                'original_width': width,
                'original_height': height
            }

        except (ValueError, TypeError, IndexError):
            return None
