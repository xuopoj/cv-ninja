"""API client for external prediction services."""

import requests
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
import json

from PIL import Image
import io

from cv_ninja.predictors.auth import AuthHandler
from cv_ninja.predictors.formats import FormatConverter, FormDataFormatConverter, BinaryFormatConverter



class PredictionClient(ABC):
    """Base class for prediction API clients.

    Different implementations handle different input/output formats:
    - FormDataPredictor: multipart/form-data requests
    - BinaryPredictor: raw binary data with query parameters
    - Base64Predictor: base64 encoded JSON (future)
    """

    def __init__(
        self,
        api_url: str,
        auth_handler: Optional[AuthHandler] = None,
        converter: Optional[FormatConverter] = None
    ):
        """Initialize the prediction client.

        Args:
            api_url: Base URL of the prediction API
            auth_handler: Optional authentication handler
            converter: Format converter to convert API responses to COCO format
        """
        self.api_url = api_url
        self.auth_handler = auth_handler
        self.converter = converter
        self.session = requests.Session()

    @abstractmethod
    def predict_from_file(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Send image file to API for prediction.

        Args:
            image_path: Path to image file
            **kwargs: Additional parameters specific to the implementation

        Returns:
            API response as dictionary

        Raises:
            requests.RequestException: If API request fails
        """
        raise NotImplementedError

    @abstractmethod
    def predict_from_bytes(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Send image bytes to API for prediction.

        Args:
            image_data: Image bytes
            **kwargs: Additional parameters specific to the implementation

        Returns:
            API response as dictionary

        Raises:
            requests.RequestException: If API request fails
        """
        raise NotImplementedError

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers if auth handler is configured.

        Returns:
            Dictionary of authentication headers
        """
        if self.auth_handler:
            return self.auth_handler.get_headers()
        return {}


class FormDataPredictor(PredictionClient):
    """Predictor that sends images as multipart/form-data.

    This format is commonly used by REST APIs that accept file uploads
    along with additional form fields (model, confidence, etc.).

    Focused on prediction only. Use ImageTiler separately for tiling support.
    """
    def __init__(self, api_url, auth_handler = None, converter = None):
        super().__init__(api_url, auth_handler, FormDataFormatConverter())

    def predict_from_file(
        self,
        image_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Send image file to API as multipart form data.

        Args:
            image_path: Path to image file
            **kwargs: Additional API-specific form fields

        Returns:
            COCO format predictions if converter is set, otherwise API response

        Raises:
            requests.RequestException: If API request fails
        """
        # Open image to get dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Read image data
        with open(image_path, 'rb') as f:
            image_data = f.read()

        result = self._make_request(
            image_data=image_data,
            image_width=width,
            image_height=height,
            **kwargs
        )

        # Convert to COCO format if converter is set
        if self.converter:
            return self.converter.to_coco(result, image_id=1, file_name=Path(image_path).name)
        return result

    def predict_from_bytes(
        self,
        image_data: bytes,
        image_id: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Send image bytes to API as multipart form data.

        Args:
            image_data: Image bytes
            image_id: Image ID for COCO format (used when converter is set)
            **kwargs: Additional API-specific form fields

        Returns:
            COCO format predictions if converter is set, otherwise API response

        Raises:
            requests.RequestException: If API request fails
        """
        # Get image dimensions from bytes
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size

        result = self._make_request(
            image_data=image_data,
            image_width=width,
            image_height=height,
            **kwargs
        )

        # Convert to COCO format if converter is set
        if self.converter:
            return self.converter.to_coco(result, image_id=image_id)
        return result

    def _make_request(
        self,
        image_data: bytes,
        image_width: int,
        image_height: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Make multipart form-data request to prediction API.

        Args:
            image_data: Image bytes
            image_width: Image width in pixels
            image_height: Image height in pixels
            **kwargs: Additional form fields

        Returns:
            Prediction results with standardized format

        Raises:
            requests.RequestException: If request fails
        """
        # Prepare multipart form data
        files = {
            'file': ('image.jpg', image_data, 'image/jpeg')
        }

        data = {
            **kwargs
        }

        # Debug: return mock data with proper metadata
        # debug_result = {
        #     'dataset_id': '1377606572385112064',
        #     'result': [
        #         {'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        #         {
        #             'Box': {'Angle': 0, 'Height': 154, 'Width': 45, 'X': 1148, 'Y': 689},
        #             'Score': 0.8662109375,
        #             'label': 'jiaza'
        #         }
        #     ],
        #     'image_width': image_width,
        #     'image_height': image_height
        # }
        # return debug_result
        headers = self._get_auth_headers()

        response = self.session.post(
            self.api_url,
            files=files,
            data=data,
            headers=headers,
            timeout=30,
            verify=False
        )

        response.raise_for_status()

        # Parse API response and add image dimensions
        # {'dataset_id': '1377606572385112064', 'result': [{'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, {'Box': {'Angle': 0, 'Height': 154, 'Width': 45, 'X': 1148, 'Y': 689}, 'Score': 0.8662109375, 'label': 'jiaza'}]}
        result = response.json()
        print(f"[DEBUG] API Response: {result}")

        # Ensure standardized format
        if 'image_width' not in result:
            result['image_width'] = image_width
        if 'image_height' not in result:
            result['image_height'] = image_height

        return result


class BinaryPredictor(PredictionClient):
    """Predictor that sends raw binary image data with query parameters.

    This format sends the image as raw bytes in the request body
    with parameters passed as URL query string. Common in simpler APIs
    that process binary uploads directly.

    Focused on prediction only. Use ImageTiler separately for tiling support.
    """

    def __init__(
        self,
        api_url: str,
        auth_handler: Optional[AuthHandler] = None,
        endpoint: str = "/upload"
    ):
        """Initialize binary predictor.

        Args:
            api_url: Base URL of the prediction API
            auth_handler: Optional authentication handler
            endpoint: API endpoint path (default: "/upload")
        """
        super().__init__(api_url, auth_handler, BinaryFormatConverter())
        self.endpoint = endpoint

    def predict_from_file(
        self,
        image_path: str,
        params: Optional[Dict[str, Any]] = None,
        image_id: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Send image file as binary data with query parameters.

        Args:
            image_path: Path to image file
            params: Query parameters to send
            image_id: Image ID for COCO format (used when converter is set)
            **kwargs: Additional query parameters

        Returns:
            COCO format predictions if converter is set, otherwise API response

        Raises:
            requests.RequestException: If API request fails
        """
        # Read image data
        with open(image_path, 'rb') as f:
            image_data = f.read()

        return self.predict_from_bytes(
            image_data=image_data,
            params=params,
            image_id=image_id,
            **kwargs
        )

    def predict_from_bytes(
        self,
        image_data: bytes,
        params: Optional[Dict[str, Any]] = None,
        image_id: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Send binary image data with query parameters.

        Args:
            image_data: Raw image bytes
            params: Query parameters to send
            image_id: Image ID for COCO format (used when converter is set)
            **kwargs: Additional query parameters

        Returns:
            COCO format predictions if converter is set, otherwise API response

        Raises:
            requests.RequestException: If API request fails
        """
        # Combine params and kwargs
        query_params = {}
        if params:
            query_params.update(params)
        query_params.update(kwargs)

        # Set headers with authentication
        headers = {
            "Content-Type": "application/octet-stream"
        }
        headers.update(self._get_auth_headers())

        # Construct full URL with endpoint
        url = self.api_url.rstrip('/') + '/' + self.endpoint.lstrip('/')

        response = self.session.post(
            url,
            data=image_data,
            params=query_params,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        # Convert to COCO format if converter is set
        if self.converter:
            return self.converter.to_coco(result, image_id=image_id)
        return result


# Legacy alias for backward compatibility
PredictionAPIClient = FormDataPredictor
