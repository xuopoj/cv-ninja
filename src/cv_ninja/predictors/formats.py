"""Annotation format converters for standardizing prediction outputs.

Uses COCO format as the internal standard exchange format for all predictors.
Each predictor implements a converter to/from COCO format.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path


class FormatConverter(ABC):
    """Base class for annotation format converters."""

    @abstractmethod
    def to_coco(self, predictions: Dict[str, Any], image_id: int = 1) -> Dict[str, Any]:
        """Convert predictor-specific format to COCO format.

        Args:
            predictions: Predictor-specific prediction results
            image_id: Image ID for COCO format (default: 1)

        Returns:
            COCO format dictionary with images, annotations, and categories
        """
        raise NotImplementedError

    @abstractmethod
    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO format back to predictor-specific format.

        Args:
            coco_data: COCO format dictionary

        Returns:
            Predictor-specific format dictionary
        """
        raise NotImplementedError


class FormDataFormatConverter(FormatConverter):
    """Converter for FormDataPredictor format.

    FormData format:
    {
        'dataset_id': '1377606572385112064',
        'result': [
            {'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {'Box': {'X': 1148, 'Y': 689, 'Width': 45, 'Height': 154, 'Angle': 0},
             'Score': 0.8662109375,
             'label': 'jiaza'}
        ],
        'image_width': 1920,
        'image_height': 1080
    }

    COCO format:
    {
        'images': [{'id': 1, 'width': 1920, 'height': 1080, 'file_name': ''}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1,
             'bbox': [1148, 689, 45, 154],
             'area': 6930, 'score': 0.866, 'iscrowd': 0}
        ],
        'categories': [{'id': 1, 'name': 'jiaza'}]
    }
    """

    def to_coco(
        self,
        predictions: Dict[str, Any],
        image_id: int = 1,
        file_name: str = ""
    ) -> Dict[str, Any]:
        """Convert FormData format to COCO format.

        Args:
            predictions: FormData format predictions
            image_id: Image ID for COCO format
            file_name: Optional file name for image entry

        Returns:
            COCO format dictionary
        """
        # Extract image info
        width = predictions.get('image_width', 0)
        height = predictions.get('image_height', 0)

        # Build COCO structure
        coco_data = {
            'images': [{
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name
            }],
            'annotations': [],
            'categories': []
        }

        # Track categories
        category_map = {}  # name -> id
        category_counter = 1

        # Process detections
        annotation_id = 1
        result_list = predictions.get('result', [])

        for item in result_list:
            # Skip RegisterMatrix
            if 'RegisterMatrix' in item:
                continue

            # Process Box detection
            if 'Box' in item:
                box = item['Box']
                label = item.get('label', 'unknown')
                score = item.get('Score', 0.0)

                # Add category if new
                if label not in category_map:
                    category_map[label] = category_counter
                    coco_data['categories'].append({
                        'id': category_counter,
                        'name': label
                    })
                    category_counter += 1

                # Convert Box format to COCO bbox [x, y, width, height]
                bbox = [
                    box['X'],
                    box['Y'],
                    box['Width'],
                    box['Height']
                ]
                area = box['Width'] * box['Height']

                # Create COCO annotation
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_map[label],
                    'bbox': bbox,
                    'area': area,
                    'score': score,
                    'iscrowd': 0
                }

                # Preserve Angle if non-zero (extended COCO)
                if box.get('Angle', 0) != 0:
                    annotation['angle'] = box['Angle']

                coco_data['annotations'].append(annotation)
                annotation_id += 1

        # Add metadata
        coco_data['metadata'] = {
            'dataset_id': predictions.get('dataset_id'),
            'num_tiles': predictions.get('num_tiles'),
            'total_detections': len(coco_data['annotations'])
        }

        return coco_data

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO format back to FormData format.

        Args:
            coco_data: COCO format dictionary

        Returns:
            FormData format dictionary
        """
        # Extract image info
        image_info = coco_data['images'][0] if coco_data.get('images') else {}
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)

        # Build category map
        category_map = {}  # id -> name
        for cat in coco_data.get('categories', []):
            category_map[cat['id']] = cat['name']

        # Build result list
        result_list = []

        # Add RegisterMatrix (identity matrix as default)
        result_list.append({
            'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        })

        # Convert annotations
        for ann in coco_data.get('annotations', []):
            bbox = ann['bbox']  # [x, y, width, height]

            # Create Box format
            box_item = {
                'Box': {
                    'X': bbox[0],
                    'Y': bbox[1],
                    'Width': bbox[2],
                    'Height': bbox[3],
                    'Angle': ann.get('angle', 0)  # Use extended angle if present
                },
                'Score': ann.get('score', 0.0),
                'label': category_map.get(ann['category_id'], 'unknown')
            }

            result_list.append(box_item)

        # Build FormData format
        formdata = {
            'dataset_id': coco_data.get('metadata', {}).get('dataset_id'),
            'result': result_list,
            'image_width': width,
            'image_height': height
        }

        # Add optional metadata
        metadata = coco_data.get('metadata', {})
        if 'num_tiles' in metadata:
            formdata['num_tiles'] = metadata['num_tiles']
        if 'total_detections' in metadata:
            formdata['total_detections'] = metadata['total_detections']

        return formdata


class BinaryFormatConverter(FormatConverter):
    """Converter for BinaryPredictor format.

    Binary format:
    {
        'result': 'success',
        'suggestion': [[
            {
                'Box': {'x1': 3466, 'y1': 1990, 'x2': 3732, 'y2': 2326},
                'Class': 'jieba',
                'Scores': 0.95458984375,
                'Picture_id': '1'
            }
        ]],
        'total_time': 0.17780423164367676
    }

    COCO format:
    {
        'images': [{'id': 1, 'width': 1920, 'height': 1080, 'file_name': ''}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1,
             'bbox': [3466, 1990, 266, 336],
             'area': 89376, 'score': 0.954, 'iscrowd': 0}
        ],
        'categories': [{'id': 1, 'name': 'jieba'}]
    }
    """

    def to_coco(
        self,
        predictions: Dict[str, Any],
        image_id: int = 1,
        file_name: str = "",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Dict[str, Any]:
        """Convert BinaryPredictor format to COCO format.

        Args:
            predictions: BinaryPredictor format predictions
            image_id: Image ID for COCO format
            file_name: Optional file name for image entry
            image_width: Image width (if known)
            image_height: Image height (if known)

        Returns:
            COCO format dictionary
        """
        # Extract suggestions (nested array structure)
        suggestions = predictions.get('suggestion', [[]])

        # Flatten nested suggestion arrays
        detections = []
        for suggestion_group in suggestions:
            if isinstance(suggestion_group, list):
                detections.extend(suggestion_group)
            else:
                detections.append(suggestion_group)

        # Infer image dimensions from bounding boxes if not provided
        if image_width is None or image_height is None:
            max_x, max_y = 0, 0
            for det in detections:
                if 'Box' in det:
                    box = det['Box']
                    max_x = max(max_x, box.get('x2', 0))
                    max_y = max(max_y, box.get('y2', 0))

            if image_width is None:
                image_width = max_x if max_x > 0 else 0
            if image_height is None:
                image_height = max_y if max_y > 0 else 0

        # Build COCO structure
        coco_data = {
            'images': [{
                'id': image_id,
                'width': image_width,
                'height': image_height,
                'file_name': file_name
            }],
            'annotations': [],
            'categories': []
        }

        # Track categories
        category_map = {}  # name -> id
        category_counter = 1

        # Process detections
        annotation_id = 1
        for det in detections:
            if 'Box' not in det:
                continue

            box = det['Box']
            label = det.get('Class', 'unknown')
            score = det.get('Scores', 0.0)

            # Add category if new
            if label not in category_map:
                category_map[label] = category_counter
                coco_data['categories'].append({
                    'id': category_counter,
                    'name': label
                })
                category_counter += 1

            # Convert Box format from {x1, y1, x2, y2} to COCO bbox [x, y, width, height]
            x1, y1 = box.get('x1', 0), box.get('y1', 0)
            x2, y2 = box.get('x2', 0), box.get('y2', 0)
            width = x2 - x1
            height = y2 - y1

            bbox = [x1, y1, width, height]
            area = width * height

            # Create COCO annotation
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_map[label],
                'bbox': bbox,
                'area': area,
                'score': score,
                'iscrowd': 0
            }

            # Preserve Picture_id if present (extended COCO)
            if 'Picture_id' in det:
                annotation['picture_id'] = det['Picture_id']

            coco_data['annotations'].append(annotation)
            annotation_id += 1

        # Add metadata
        coco_data['metadata'] = {
            'result_status': predictions.get('result'),
            'total_time': predictions.get('total_time'),
            'total_detections': len(coco_data['annotations'])
        }

        return coco_data

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO format back to BinaryPredictor format.

        Args:
            coco_data: COCO format dictionary

        Returns:
            BinaryPredictor format dictionary
        """
        # Build category map
        category_map = {}  # id -> name
        for cat in coco_data.get('categories', []):
            category_map[cat['id']] = cat['name']

        # Build detections list
        detections = []

        for ann in coco_data.get('annotations', []):
            bbox = ann['bbox']  # [x, y, width, height]

            # Convert COCO bbox to {x1, y1, x2, y2} format
            x1, y1 = bbox[0], bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]

            detection = {
                'Box': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                },
                'Class': category_map.get(ann['category_id'], 'unknown'),
                'Scores': ann.get('score', 0.0)
            }

            # Preserve picture_id if present
            if 'picture_id' in ann:
                detection['Picture_id'] = ann['picture_id']

            detections.append(detection)

        # Build BinaryPredictor format with nested suggestion structure
        metadata = coco_data.get('metadata', {})
        binary_format = {
            'result': metadata.get('result_status', 'success'),
            'suggestion': [detections],  # Wrap in nested array
        }

        # Add optional metadata
        if 'total_time' in metadata:
            binary_format['total_time'] = metadata['total_time']

        return binary_format


class LabelStudioFormatConverter(FormatConverter):
    """Converter for Label Studio annotation format.

    Converts COCO format to/from Label Studio JSON format with scores
    visible in the annotations metadata.

    Label Studio format:
    {
        'data': {
            'image': 'path/to/image.jpg',
            'filename': 'image.jpg',
            'label': 'jieba, jiaza',
            'review_label': 'FN',
            'target_class': 'jieba'
        },
        'annotations': [{
            'result': [{
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'x': 49.17,  # percentage
                    'y': 96.4,   # percentage
                    'width': 2.15,
                    'height': 3.53,
                    'rectanglelabels': ['jieba']
                },
                'original_width': 4096,
                'original_height': 3000,
                'score': 0.712890625
            }]
        }]
    }
    """

    def __init__(self, prefix: str = "", output_mode: str = "annotations"):
        """Initialize Label Studio converter.

        Args:
            prefix: URL prefix to prepend to image paths (e.g., "http://localhost:8080/data/")
            output_mode: Use 'annotations' or 'predictions' field (default: 'annotations')
        """
        self.prefix = prefix
        self.output_mode = output_mode

    @staticmethod
    def _parse_filename_metadata(filename: str) -> tuple:
        """Parse filename to extract metadata.

        Parses filenames in the format: {REVIEW_LABEL}_{TARGET_CLASS}_rest_of_name.ext
        where REVIEW_LABEL can be FN (False Negative) or FP (False Positive).

        Args:
            filename: Full filename or path

        Returns:
            Tuple of (basename, review_label, target_class)

        Examples:
            "FN_jieba_image001.jpg" -> ("FN_jieba_image001.jpg", "FN", "jieba")
            "FP_maobian_test.png" -> ("FP_maobian_test.png", "FP", "maobian")
            "/path/to/image.jpg" -> ("image.jpg", None, None)
        """
        basename = Path(filename).name

        review_label = None
        target_class = None

        if basename.startswith("FN_") or basename.startswith("FP_"):
            parts = basename.split("_")
            if len(parts) >= 2:
                review_label = parts[0]  # FN or FP
                target_class = parts[1]  # defect type

        return basename, review_label, target_class

    def to_coco(self, predictions: Dict[str, Any], image_id: int = 1) -> Dict[str, Any]:
        """Convert Label Studio format to COCO format.

        Args:
            predictions: Label Studio format predictions
            image_id: Image ID for COCO format

        Returns:
            COCO format dictionary
        """
        # Extract image info from data
        data = predictions.get('data', {})
        image_path = data.get('image', '')

        # Get image dimensions from first annotation if available
        annotations_list = predictions.get('annotations', [])
        if annotations_list and annotations_list[0].get('result'):
            first_result = annotations_list[0]['result'][0]
            image_width = first_result.get('original_width', 0)
            image_height = first_result.get('original_height', 0)
        else:
            image_width = 0
            image_height = 0

        # Build COCO structure
        coco_data = {
            'images': [{
                'id': image_id,
                'width': image_width,
                'height': image_height,
                'file_name': Path(image_path).name
            }],
            'annotations': [],
            'categories': []
        }

        # Track categories
        category_map = {}  # name -> id
        category_counter = 1
        annotation_id = 1

        # Process all annotations
        for annotation_group in annotations_list:
            for result in annotation_group.get('result', []):
                if result.get('type') != 'rectanglelabels':
                    continue

                value = result.get('value', {})
                labels = value.get('rectanglelabels', [])

                if not labels:
                    continue

                label = labels[0]  # Take first label

                # Add category if new
                if label not in category_map:
                    category_map[label] = category_counter
                    coco_data['categories'].append({
                        'id': category_counter,
                        'name': label
                    })
                    category_counter += 1

                # Convert percentage to pixel coordinates
                x_percent = value.get('x', 0)
                y_percent = value.get('y', 0)
                w_percent = value.get('width', 0)
                h_percent = value.get('height', 0)

                x = (x_percent / 100) * image_width
                y = (y_percent / 100) * image_height
                w = (w_percent / 100) * image_width
                h = (h_percent / 100) * image_height

                bbox = [x, y, w, h]
                area = w * h

                # Extract score from result
                score = result.get('score', 0.0)

                # Create COCO annotation
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_map[label],
                    'bbox': bbox,
                    'area': area,
                    'score': score,
                    'iscrowd': 0
                })
                annotation_id += 1

        # Add metadata
        coco_data['metadata'] = {
            'review_label': data.get('review_label'),
            'target_class': data.get('target_class'),
            'total_detections': len(coco_data['annotations'])
        }

        return coco_data

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO format to Label Studio format.

        Args:
            coco_data: COCO format dictionary

        Returns:
            Label Studio format dictionary with scores in meta
        """
        # Extract image info
        image_info = coco_data.get('images', [{}])[0]
        image_width = image_info.get('width', 0)
        image_height = image_info.get('height', 0)
        image_path = image_info.get('file_name', 'image.jpg')

        # Parse filename metadata
        image_name, review_label, target_class = self._parse_filename_metadata(image_path)

        # Build category map
        category_map = {}  # id -> name
        for cat in coco_data.get('categories', []):
            category_map[cat['id']] = cat['name']

        # Build annotations
        result_list = []
        labels_set = set()

        for ann in coco_data.get('annotations', []):
            bbox = ann['bbox']  # [x, y, width, height]
            category_name = category_map.get(ann['category_id'], 'unknown')
            score = ann.get('score', 0.0)

            labels_set.add(category_name)

            # Convert pixel coords to Label Studio percentage format
            x_percent = (bbox[0] / image_width) * 100 if image_width > 0 else 0
            y_percent = (bbox[1] / image_height) * 100 if image_height > 0 else 0
            w_percent = (bbox[2] / image_width) * 100 if image_width > 0 else 0
            h_percent = (bbox[3] / image_height) * 100 if image_height > 0 else 0

            result_list.append({
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'x': x_percent,
                    'y': y_percent,
                    'width': w_percent,
                    'height': h_percent,
                    'rectanglelabels': [category_name]
                },
                'original_width': image_width,
                'original_height': image_height,
                'score': score
            })

        # Build data dict
        data = {
            'image': f'{self.prefix}{image_name}',
            'filename': image_name,
            'label': ', '.join(sorted(labels_set))
        }

        # Add metadata from COCO if present
        metadata = coco_data.get('metadata', {})
        if review_label or metadata.get('review_label'):
            data['review_label'] = review_label or metadata.get('review_label')
        if target_class or metadata.get('target_class'):
            data['target_class'] = target_class or metadata.get('target_class')

        result_data = {
            'data': data,
        }

        # Use annotations or predictions based on output_mode
        if self.output_mode == "predictions":
            result_data["predictions"] = [{'result': result_list}] if result_list else []
        else:
            result_data["annotations"] = [{'result': result_list}] if result_list else []

        return result_data
