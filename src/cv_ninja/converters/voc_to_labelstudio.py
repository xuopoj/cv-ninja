"""Convert Pascal VOC annotations to Label Studio JSON format."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from cv_ninja.converters.base import BaseConverter
from cv_ninja.utils.exceptions import ConversionError, ValidationError


class VOCToLabelStudioConverter(BaseConverter):
    """Converter for Pascal VOC to Label Studio JSON format.

    Converts Pascal VOC XML annotation files to Label Studio's JSON format
    with bounding box annotations as percentage coordinates.
    """

    def __init__(self, prefix: str = '', image_dir: Optional[str] = None):
        """Initialize the converter.

        Args:
            prefix: URL prefix to prepend to image paths (e.g., '/data/images/')
            image_dir: Directory containing image files. If provided, dimensions
                      will be read from actual images instead of XML files.
        """
        self.prefix = prefix
        self.image_dir = Path(image_dir) if image_dir else None

    @property
    def supported_formats(self) -> Tuple[str, str]:
        """Return supported format conversion."""
        return ('pascal_voc', 'labelstudio')

    def validate_input(self, input_path: str) -> bool:
        """Validate that input directory exists and contains XML files.

        Args:
            input_path: Directory path containing Pascal VOC XML files

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

        xml_files = list(path.glob('*.xml'))
        if not xml_files:
            raise ValidationError(f"No XML files found in directory: {input_path}")

        return True

    def convert(self, input_dir: str, output_path: str, **kwargs) -> None:
        """Convert all VOC XML files in directory to Label Studio format.

        Args:
            input_dir: Directory containing Pascal VOC XML files
            output_path: Output JSON file path
            **kwargs: Additional options (unused)

        Raises:
            ConversionError: If conversion fails
        """
        try:
            self.validate_input(input_dir)
            voc_path = Path(input_dir)
            tasks = []

            for xml_file in sorted(voc_path.glob('*.xml')):
                try:
                    filename, width, height, annotations = self._parse_voc_xml(
                        xml_file, self.image_dir
                    )

                    # Extract unique labels for column display
                    labels = []
                    if annotations:
                        labels = list(set(
                            ann['value']['rectanglelabels'][0]
                            for ann in annotations
                        ))

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
                        f"Failed to convert {xml_file.name}: {str(e)}"
                    ) from e

            # Write output JSON
            with open(output_path, 'w') as f:
                json.dump(tasks, f, indent=2)

        except ValidationError:
            raise
        except Exception as e:
            raise ConversionError(f"Conversion failed: {str(e)}") from e

    def _parse_voc_xml(
        self, xml_path: Path, image_dir: Optional[Path] = None
    ) -> Tuple[str, int, int, List[dict]]:
        """Parse Pascal VOC XML file and extract annotations.

        Args:
            xml_path: Path to Pascal VOC XML file
            image_dir: Optional directory containing image files. If provided,
                      dimensions will be read from actual images.

        Returns:
            Tuple of (filename, width, height, annotations)

        Raises:
            ConversionError: If parsing fails
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Use actual XML filename instead of reading from XML content
            filename = xml_path.stem + '.jpg'

            # Read dimensions from actual image file if image_dir provided
            if image_dir is not None:
                image_path = image_dir / filename
                if image_path.exists():
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                    except Exception as e:
                        raise ConversionError(
                            f"Failed to read image dimensions from {image_path.name}: {str(e)}"
                        ) from e
                else:
                    raise ConversionError(
                        f"Image file not found: {image_path}"
                    )
            else:
                # Fall back to reading dimensions from XML
                size = root.find('size')
                if size is None:
                    raise ConversionError(
                        f"Missing <size> element in {xml_path.name}"
                    )

                width_elem = size.find('width')
                height_elem = size.find('height')

                if width_elem is None or height_elem is None:
                    raise ConversionError(
                        f"Missing width/height in <size> element in {xml_path.name}"
                    )

                try:
                    width = int(width_elem.text)
                    height = int(height_elem.text)
                except (ValueError, AttributeError) as e:
                    raise ConversionError(
                        f"Invalid width/height values in {xml_path.name}"
                    ) from e

                if width <= 0 or height <= 0:
                    raise ConversionError(
                        f"Invalid dimensions ({width}x{height}) in {xml_path.name}"
                    )

            annotations = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None or not name_elem.text:
                    continue  # Skip objects without name

                name = name_elem.text
                bbox = obj.find('bndbox')

                if bbox is None:
                    continue  # Skip objects without bounding box

                try:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                except (AttributeError, ValueError) as e:
                    # Skip invalid bounding boxes
                    continue

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

        except ET.ParseError as e:
            raise ConversionError(
                f"Failed to parse XML file {xml_path.name}: {str(e)}"
            ) from e
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Unexpected error parsing {xml_path.name}: {str(e)}"
            ) from e
