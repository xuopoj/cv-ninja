"""Image tiling utilities for processing large images in patches.

ImageTiler orchestrates tiling operations by calling a predictor
on each tile and combining the results.
"""

from typing import List, Tuple, Dict, Any, Callable
from PIL import Image
import io


class ImageTiler:
    """Handle splitting large images into tiles and combining prediction results.

    Used when images are larger than the model's maximum input size.
    Tiles overlap to ensure objects near tile boundaries are detected.

    The tiler orchestrates the prediction process by:
    1. Splitting the image into tiles
    2. Calling the predictor on each tile
    3. Converting results to COCO format
    4. Combining and applying NMS
    5. Converting back to original format
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (1386, 1516),
        overlap: int = 32,
        iou_threshold: float = 0.5
    ):
        """Initialize image tiler.

        Args:
            tile_size: Maximum tile dimensions (width, height)
            overlap: Overlap between tiles in pixels (default: 32)
            iou_threshold: IoU threshold for NMS (default: 0.5)
        """
        self.tile_width, self.tile_height = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold

    def needs_tiling(self, image: Image.Image) -> bool:
        """Check if image needs to be tiled.

        Args:
            image: PIL Image

        Returns:
            True if image exceeds tile size
        """
        width, height = image.size
        return width > self.tile_width or height > self.tile_height

    def split_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Split image into overlapping tiles.

        Args:
            image: PIL Image to split

        Returns:
            List of tile dictionaries containing:
                - image: PIL Image (tile)
                - x_offset: X position in original image
                - y_offset: Y position in original image
                - tile_index: Sequential tile number
        """
        width, height = image.size
        tiles = []
        tile_index = 0

        # Calculate stride (step size between tiles)
        stride_x = self.tile_width - self.overlap
        stride_y = self.tile_height - self.overlap

        y = 0
        while y < height:
            x = 0
            while x < width:
                # Calculate tile boundaries
                x_end = min(x + self.tile_width, width)
                y_end = min(y + self.tile_height, height)

                # Adjust start position if we're at the edge
                x_start = max(0, x_end - self.tile_width) if x_end == width else x
                y_start = max(0, y_end - self.tile_height) if y_end == height else y

                # Crop tile
                tile = image.crop((x_start, y_start, x_end, y_end))

                tiles.append({
                    'image': tile,
                    'x_offset': x_start,
                    'y_offset': y_start,
                    'tile_index': tile_index,
                    'tile_width': x_end - x_start,
                    'tile_height': y_end - y_start
                })

                tile_index += 1
                x += stride_x

                if x >= width:
                    break

            y += stride_y

            if y >= height:
                break

        return tiles

    def tile_to_bytes(self, tile: Image.Image) -> bytes:
        """Convert tile image to bytes.

        Args:
            tile: PIL Image

        Returns:
            Image bytes in JPEG format
        """
        buffer = io.BytesIO()
        tile.save(buffer, format='JPEG')
        return buffer.getvalue()

    def predict_tiled(
        self,
        predictor: Any,  # PredictionClient instance (must return COCO format)
        image: Image.Image,
        **predict_kwargs
    ) -> Dict[str, Any]:
        """Predict on large image using tiling.

        Orchestrates the entire tiling workflow:
        1. Split image into tiles
        2. Call predictor on each tile (predictor returns COCO format)
        3. Combine and apply NMS
        4. Return combined COCO format

        Args:
            predictor: PredictionClient instance (must have converter set to return COCO)
            image: PIL Image to process
            **predict_kwargs: Keyword arguments to pass to predictor.predict_from_bytes()

        Returns:
            Combined prediction results in COCO format
        """
        # Split image into tiles
        tiles = self.split_image(image)

        # Predict on each tile (predictor already returns COCO)
        tile_predictions = []
        for tile_info in tiles:
            tile_bytes = self.tile_to_bytes(tile_info['image'])

            # Call predictor (returns COCO format if converter is set)
            coco_data = predictor.predict_from_bytes(
                tile_bytes,
                image_id=tile_info['tile_index'],
                **predict_kwargs
            )

            # Store with tile metadata
            tile_predictions.append({
                'coco_data': coco_data,
                'x_offset': tile_info['x_offset'],
                'y_offset': tile_info['y_offset']
            })

        # Combine predictions in COCO format
        combined_coco = self.combine_predictions(
            tile_predictions,
            original_size=image.size,
            iou_threshold=self.iou_threshold
        )

        return combined_coco

    def combine_predictions(
        self,
        tile_predictions: List[Dict[str, Any]],
        original_size: Tuple[int, int],
        iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Combine predictions from multiple tiles into single result.

        Uses COCO format as standard exchange format:
        {
            'images': [{'id': 1, 'width': w, 'height': h}],
            'annotations': [{'id': 1, 'bbox': [x, y, w, h], 'score': 0.9, ...}],
            'categories': [{'id': 1, 'name': 'class'}]
        }

        Args:
            tile_predictions: List of COCO format predictions with tile metadata
            original_size: Original image size (width, height)
            iou_threshold: IoU threshold for NMS (default: 0.5)

        Returns:
            Combined COCO format prediction result
        """
        all_annotations = []
        all_categories = {}  # name -> category dict
        metadata = {}
        annotation_id = 1

        # Transform coordinates from tile space to original image space
        for pred in tile_predictions:
            x_offset = pred['x_offset']
            y_offset = pred['y_offset']
            coco_data = pred.get('coco_data', {})

            # Collect categories
            for cat in coco_data.get('categories', []):
                cat_name = cat['name']
                if cat_name not in all_categories:
                    all_categories[cat_name] = cat.copy()

            # Collect and preserve metadata from first tile
            if not metadata and 'metadata' in coco_data:
                metadata = coco_data['metadata'].copy()

            # Process annotations with coordinate adjustment
            for ann in coco_data.get('annotations', []):
                # Adjust bbox coordinates: [x, y, width, height]
                bbox = ann['bbox'].copy() if isinstance(ann['bbox'], list) else list(ann['bbox'])
                bbox[0] += x_offset  # Adjust X
                bbox[1] += y_offset  # Adjust Y

                # Create adjusted annotation
                adjusted_ann = ann.copy()
                adjusted_ann['bbox'] = bbox
                adjusted_ann['id'] = annotation_id  # Will be reassigned after NMS
                all_annotations.append(adjusted_ann)
                annotation_id += 1

        # Apply Non-Maximum Suppression to remove duplicate detections at boundaries
        if all_annotations:
            all_annotations = self._apply_nms_coco_format(all_annotations, iou_threshold)

        # Reassign annotation IDs after NMS
        for idx, ann in enumerate(all_annotations, start=1):
            ann['id'] = idx

        # Reassign category IDs (make them sequential)
        category_list = []
        category_name_to_id = {}
        for idx, (cat_name, cat_dict) in enumerate(sorted(all_categories.items()), start=1):
            category_name_to_id[cat_name] = idx
            category_list.append({
                'id': idx,
                'name': cat_name
            })

        # Update category IDs in annotations
        for ann in all_annotations:
            # Find category name from old categories
            old_cat_id = ann['category_id']
            cat_name = None
            for name, cat in all_categories.items():
                if cat['id'] == old_cat_id:
                    cat_name = name
                    break
            if cat_name:
                ann['category_id'] = category_name_to_id[cat_name]

        # Build combined COCO result
        combined = {
            'images': [{
                'id': 1,
                'width': original_size[0],
                'height': original_size[1],
                'file_name': ''
            }],
            'annotations': all_annotations,
            'categories': category_list,
            'metadata': {
                **metadata,
                'num_tiles': len(tile_predictions),
                'total_detections': len(all_annotations)
            }
        }

        return combined

    def _apply_nms_coco_format(
        self,
        annotations: List[Dict[str, Any]],
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression for COCO format annotations.

        Args:
            annotations: List of COCO annotation dictionaries with 'bbox' and 'score'
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of annotations
        """
        if not annotations:
            return []

        # Sort by score (descending)
        sorted_annotations = sorted(
            annotations,
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        keep = []

        while sorted_annotations:
            # Take the annotation with highest score
            best = sorted_annotations.pop(0)
            keep.append(best)

            # Remove annotations with high IoU with the best annotation
            sorted_annotations = [
                ann for ann in sorted_annotations
                if self._calculate_iou_coco_format(best.get('bbox'), ann.get('bbox')) < iou_threshold
            ]

        return keep

    def _apply_nms_box_format(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression for Box format detections.

        Legacy method for FormData Box format.

        Args:
            detections: List of detection dictionaries with 'Box' and 'Score'
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by Score (descending)
        sorted_detections = sorted(
            detections,
            key=lambda x: x.get('Score', 0),
            reverse=True
        )

        keep = []

        while sorted_detections:
            # Take the detection with highest score
            best = sorted_detections.pop(0)
            keep.append(best)

            # Remove detections with high IoU with the best detection
            sorted_detections = [
                det for det in sorted_detections
                if self._calculate_iou_box_format(best.get('Box'), det.get('Box')) < iou_threshold
            ]

        return keep

    def _apply_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove duplicate detections.

        Legacy method for standard bbox format.

        Args:
            detections: List of detection dictionaries with 'bbox' and 'score'
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence score (descending)
        sorted_detections = sorted(
            detections,
            key=lambda x: x.get('score', x.get('confidence', 0)),
            reverse=True
        )

        keep = []

        while sorted_detections:
            # Take the detection with highest confidence
            best = sorted_detections.pop(0)
            keep.append(best)

            # Remove detections with high IoU with the best detection
            sorted_detections = [
                det for det in sorted_detections
                if self._calculate_iou(best.get('bbox'), det.get('bbox')) < iou_threshold
            ]

        return keep

    def _calculate_iou_coco_format(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union for COCO bbox format.

        Args:
            bbox1: First bbox [x, y, width, height]
            bbox2: Second bbox [x, y, width, height]

        Returns:
            IoU value between 0 and 1
        """
        if not bbox1 or not bbox2:
            return 0.0

        # COCO bbox is [x, y, width, height]
        # Convert to [x1, y1, x2, y2]
        x1_1, y1_1 = bbox1[0], bbox1[1]
        x2_1, y2_1 = x1_1 + bbox1[2], y1_1 + bbox1[3]

        x1_2, y1_2 = bbox2[0], bbox2[1]
        x2_2, y2_2 = x1_2 + bbox2[2], y1_2 + bbox2[3]

        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_iou_box_format(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Calculate Intersection over Union for Box format.

        Legacy method for FormData Box format.

        Args:
            box1: First box {'X': x, 'Y': y, 'Width': w, 'Height': h, 'Angle': a}
            box2: Second box {'X': x, 'Y': y, 'Width': w, 'Height': h, 'Angle': a}

        Returns:
            IoU value between 0 and 1
        """
        if not box1 or not box2:
            return 0.0

        # For simplicity, ignore Angle and treat as axis-aligned boxes
        # Convert to [x1, y1, x2, y2]
        x1_1 = box1['X']
        y1_1 = box1['Y']
        x2_1 = x1_1 + box1['Width']
        y2_1 = y1_1 + box1['Height']

        x1_2 = box2['X']
        y1_2 = box2['Y']
        x2_2 = x1_2 + box2['Width']
        y2_2 = y1_2 + box2['Height']

        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        area1 = box1['Width'] * box1['Height']
        area2 = box2['Width'] * box2['Height']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes.

        Legacy method for standard bbox format.

        Args:
            bbox1: First bbox [x, y, width, height]
            bbox2: Second bbox [x, y, width, height]

        Returns:
            IoU value between 0 and 1
        """
        if not bbox1 or not bbox2:
            return 0.0

        # Convert to [x1, y1, x2, y2]
        x1_1, y1_1 = bbox1[0], bbox1[1]
        x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]

        x1_2, y1_2 = bbox2[0], bbox2[1]
        x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
