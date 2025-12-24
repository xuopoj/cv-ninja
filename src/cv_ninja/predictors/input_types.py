"""Prediction input types and request models."""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BasePrediction(ABC):
    """Abstract base class for prediction inputs."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate prediction input.

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        pass


@dataclass
class Prediction(BasePrediction):
    """Simple prediction object with minimal metadata.

    Used for programmatic API calls with basic metadata.
    """

    def validate(self) -> bool:
        """Validate prediction."""
        return True


@dataclass
class ImagePrediction(BasePrediction):
    """Prediction with local image file path.

    Supports local file path processing with full metadata.
    Used for single image predictions.
    """

    image_path: str
    output_format: str = "labelstudio"

    def validate(self) -> bool:
        """Validate image path and parameters."""
        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {self.image_path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        return True


@dataclass
class Base64ImagePrediction(BasePrediction):
    """Prediction with base64-encoded image data.

    Used for HTTP JSON payloads with embedded image data.
    Supports REST API with JSON content-type.
    """

    image_data: str
    image_format: str = "jpeg"
    output_format: str = "labelstudio"

    def validate(self) -> bool:
        """Validate base64 data and parameters."""
        if not self.image_data:
            raise ValueError("image_data is required")
        if self.image_format.lower() not in {"jpeg", "jpg", "png", "bmp", "gif"}:
            raise ValueError(f"Unsupported image format: {self.image_format}")
        try:
            base64.b64decode(self.image_data, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {str(e)}")
        return True

    def decode_image(self) -> bytes:
        """Decode base64 image data to bytes.

        Returns:
            Decoded image bytes
        """
        return base64.b64decode(self.image_data)


@dataclass
class URLImagePrediction(BasePrediction):
    """Prediction with image URL reference.

    Used for HTTP requests with URL-based image reference.
    Supports REST API with JSON content-type.
    """

    image_url: str
    output_format: str = "labelstudio"
    download_image: bool = False

    def validate(self) -> bool:
        """Validate URL format and parameters."""
        if not self.image_url:
            raise ValueError("image_url is required")
        if not self.image_url.startswith(("http://", "https://")):
            raise ValueError("image_url must be a valid HTTP/HTTPS URL")
        return True


@dataclass
class BatchImagePrediction(BasePrediction):
    """Batch prediction for directory of images.

    Processes all images in a directory with consistent parameters.
    Used for batch CLI operations.
    """

    image_dir: str
    output_format: str = "labelstudio"
    recursive: bool = False
    supported_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".gif"])

    def validate(self) -> bool:
        """Validate directory and parameters."""
        path = Path(self.image_dir)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {self.image_dir}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {self.image_dir}")
        return True

    def get_images(self) -> List[Path]:
        """Get all image files in directory.

        Returns:
            Sorted list of image file paths
        """
        pattern = "**/*" if self.recursive else "*"
        images = []
        path = Path(self.image_dir)
        for ext in self.supported_extensions:
            images.extend(path.glob(f"{pattern}{ext}"))
            images.extend(path.glob(f"{pattern}{ext.upper()}"))
        return sorted(list(set(images)))
