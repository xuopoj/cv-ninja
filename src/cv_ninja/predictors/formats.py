"""Annotation format converters for standardizing prediction outputs.

Uses COCO format as the internal standard exchange format for all predictors.
Each predictor implements a converter to/from COCO format.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


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

    To be implemented when BinaryPredictor format is known.
    """

    def to_coco(self, predictions: Dict[str, Any], image_id: int = 1) -> Dict[str, Any]:
        """Convert BinaryPredictor format to COCO format.

        Args:
            predictions: BinaryPredictor format predictions
            image_id: Image ID for COCO format

        Returns:
            COCO format dictionary
        """
        raise NotImplementedError("BinaryPredictor format not yet defined")

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert COCO format back to BinaryPredictor format.

        Args:
            coco_data: COCO format dictionary

        Returns:
            BinaryPredictor format dictionary
        """
        raise NotImplementedError("BinaryPredictor format not yet defined")
