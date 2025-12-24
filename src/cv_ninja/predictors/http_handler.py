"""HTTP request and response handling for predictions."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from cv_ninja.predictors.input_types import (
    Base64ImagePrediction,
    ImagePrediction,
    URLImagePrediction,
)


class HTTPRequestParser:
    """Parse and validate HTTP requests for prediction API."""

    @staticmethod
    def parse_json_request(
        payload: Dict[str, Any]
    ) -> Union[Base64ImagePrediction, URLImagePrediction]:
        """Parse JSON REST API request.

        Supports two formats:
        1. Base64 image in request body:
           {
               "image_data": "base64_string",
               "image_format": "jpeg",
               "output_format": "labelstudio"
           }

        2. Image URL reference:
           {
               "image_url": "https://example.com/image.jpg",
               "output_format": "labelstudio"
           }

        Args:
            payload: JSON request body as dictionary

        Returns:
            Appropriate prediction input object

        Raises:
            ValueError: If request format is invalid
        """
        if "image_data" in payload:
            return Base64ImagePrediction(
                image_data=payload["image_data"],
                image_format=payload.get("image_format", "jpeg"),
                output_format=payload.get("output_format", "labelstudio"),
            )
        elif "image_url" in payload:
            return URLImagePrediction(
                image_url=payload["image_url"],
                output_format=payload.get("output_format", "labelstudio"),
            )
        else:
            raise ValueError("Request must include 'image_data' or 'image_url'")

    @staticmethod
    def parse_multipart_request(
        files: Dict[str, Any], form_data: Dict[str, str]
    ) -> ImagePrediction:
        """Parse multipart form data request.

        Form fields:
        - output_format (optional, default 'labelstudio')

        Files:
        - image (required): Image file upload

        Args:
            files: Uploaded files dict (file field -> file object)
            form_data: Form fields dict

        Returns:
            ImagePrediction object

        Raises:
            ValueError: If required fields missing or invalid
        """
        if "image" not in files:
            raise ValueError("'image' file is required in multipart request")

        # Save uploaded file to temporary location
        image_file = files["image"]
        temp_path = Path("/tmp") / image_file.filename
        image_file.save(str(temp_path))

        return ImagePrediction(
            image_path=str(temp_path),
            output_format=form_data.get("output_format", "labelstudio"),
        )


class HTTPResponseFormatter:
    """Format prediction results for HTTP responses."""

    @staticmethod
    def format_json_response(
        predictions: Dict[str, Any], output_format: str = "labelstudio"
    ) -> Dict[str, Any]:
        """Format predictions as JSON response.

        Returns standardized JSON response with predictions converted
        to specified output format.

        Args:
            predictions: Raw predictions from model
            output_format: Target annotation format (labelstudio, voc, coco)

        Returns:
            JSON-serializable response dict with fields:
                - success: bool
                - predictions: Formatted annotations
                - metadata: Image/result metadata
        """
        return {
            "success": True,
            "predictions": predictions.get("detections", []),
            "metadata": {
                "image_width": predictions.get("image_width"),
                "image_height": predictions.get("image_height"),
                "output_format": output_format,
            },
        }

    @staticmethod
    def format_error_response(
        error: str, status_code: int = 400, details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format error response.

        Args:
            error: Error message
            status_code: HTTP status code
            details: Additional error details

        Returns:
            Error response dict
        """
        return {
            "success": False,
            "error": error,
            "status_code": status_code,
            "details": details,
        }
