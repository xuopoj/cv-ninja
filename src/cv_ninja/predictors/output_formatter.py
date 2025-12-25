"""Convert predictions to various annotation formats."""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class PredictionOutputFormatter:
    """Convert raw predictions to various annotation formats.

    Integrates with existing BaseConverter pattern to support
    VOC, Label Studio, COCO, and other formats.
    """

    @staticmethod
    def _parse_filename_metadata(filename: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse filename to extract base name and optional metadata.

        Parses filenames in the format: {REVIEW_LABEL}_{TARGET_CLASS}_rest_of_name.ext
        where REVIEW_LABEL can be FN (False Negative) or FP (False Positive).

        Args:
            filename: Full filename or path

        Returns:
            Tuple of (basename, review_label, target_class)
            - basename: Just the filename without path
            - review_label: FN or FP if present, None otherwise
            - target_class: Defect type if present, None otherwise

        Examples:
            "FN_jieba_image001.jpg" -> ("FN_jieba_image001.jpg", "FN", "jieba")
            "FP_maobian_test.png" -> ("FP_maobian_test.png", "FP", "maobian")
            "/path/to/image.jpg" -> ("image.jpg", None, None)
        """
        # Extract just the filename (no path)
        basename = Path(filename).name

        # Check if filename starts with FN_ or FP_
        review_label = None
        target_class = None

        if basename.startswith("FN_") or basename.startswith("FP_"):
            parts = basename.split("_")
            if len(parts) >= 2:
                review_label = parts[0]  # FN or FP
                target_class = parts[1]  # defect type (jieba, maobian, etc.)

        return basename, review_label, target_class

    @staticmethod
    def to_labelstudio(predictions: Dict[str, Any], prefix: str = "", output_mode: str = "annotations") -> Dict[str, Any]:
        """Convert predictions to Label Studio format.

        Args:
            predictions: Raw predictions in COCO format with:
                - images: List of image info
                - annotations: List of detection annotations
                - categories: List of category definitions
            prefix: URL prefix for image path
            output_mode: Use 'annotations' or 'predictions' field (default: 'annotations')

        Returns:
            Label Studio task object with annotations or predictions
        """
        # Extract image info
        image_info = predictions.get("images", [{}])[0]
        image_width = image_info.get("width", 4096)
        image_height = image_info.get("height", 3000)
        image_path = image_info.get("file_name", "image.jpg")

        # Parse filename metadata (extract basename, review_label, target_class)
        image_name, review_label, target_class = PredictionOutputFormatter._parse_filename_metadata(image_path)

        # Create category mapping
        categories = {cat["id"]: cat["name"] for cat in predictions.get("categories", [])}

        annotations = []
        labels = set()

        for detection in predictions.get("annotations", []):
            x, y, w, h = detection["bbox"]
            category_name = categories.get(detection["category_id"], "unknown")
            score = detection.get("score", 0.0)

            # Convert pixel coords to Label Studio percentage format
            x_percent = (x / image_width) * 100
            y_percent = (y / image_height) * 100
            w_percent = (w / image_width) * 100
            h_percent = (h / image_height) * 100

            annotations.append(
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": w_percent,
                        "height": h_percent,
                        "rectanglelabels": [category_name],
                    },
                    "original_width": image_width,
                    "original_height": image_height,
                    "score": score
                }
            )
            labels.add(category_name)

        # Build data dict with optional review metadata
        data = {
            "image": f'{prefix}{image_name}',
            "filename": image_name,
            "label": ", ".join(sorted(labels)),
        }

        # Add review metadata if present in filename
        if review_label:
            data["review_label"] = review_label
        if target_class:
            data["target_class"] = target_class

        result_data = {
            "data": data,
        }

        # Use annotations or predictions based on output_mode
        if output_mode == "predictions":
            result_data["predictions"] = [{"result": annotations}] if annotations else []
        else:
            result_data["annotations"] = [{"result": annotations}] if annotations else []

        return result_data

    @staticmethod
    def to_voc(predictions: Dict[str, Any], image_name: str) -> str:
        """Convert predictions to Pascal VOC XML format.

        Args:
            predictions: Raw predictions
            image_name: Name of the image file

        Returns:
            VOC XML string
        """
        image_width = predictions["image_width"]
        image_height = predictions["image_height"]

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<annotation>",
            f"  <filename>{image_name}</filename>",
            "  <size>",
            f"    <width>{image_width}</width>",
            f"    <height>{image_height}</height>",
            "    <depth>3</depth>",
            "  </size>",
        ]

        for detection in predictions.get("detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            xml_parts.extend(
                [
                    "  <object>",
                    f"    <name>{detection['class']}</name>",
                    f"    <confidence>{detection['confidence']:.4f}</confidence>",
                    "    <bndbox>",
                    f"      <xmin>{int(x1)}</xmin>",
                    f"      <ymin>{int(y1)}</ymin>",
                    f"      <xmax>{int(x2)}</xmax>",
                    f"      <ymax>{int(y2)}</ymax>",
                    "    </bndbox>",
                    "  </object>",
                ]
            )

        xml_parts.append("</annotation>")
        return "\n".join(xml_parts)

    @staticmethod
    def to_coco(
        predictions: Dict[str, Any],
        image_id: int,
        category_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to COCO format.

        Args:
            predictions: Raw predictions
            image_id: Image ID in COCO dataset
            category_map: Mapping of class names to COCO category IDs

        Returns:
            COCO format annotations list
        """
        if category_map is None:
            category_map = {}

        annotations = []
        annotation_id = 1

        for detection in predictions.get("detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            width = x2 - x1
            height = y2 - y1

            class_name = detection["class"]
            category_id = category_map.get(class_name, 1)

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

        return annotations
