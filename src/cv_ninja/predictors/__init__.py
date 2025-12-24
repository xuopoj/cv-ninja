"""Prediction module for CV Ninja.

This module provides functionality for running predictions using external
computer vision model APIs and converting results to various annotation formats.
"""

from cv_ninja.predictors.auth import APIKeyAuth, IAMTokenAuth
from cv_ninja.predictors.base import PredictionAPIClient, FormDataPredictor, BinaryPredictor
from cv_ninja.predictors.config import PredictionConfig
from cv_ninja.predictors.formats import FormDataFormatConverter, BinaryFormatConverter
from cv_ninja.predictors.http_handler import HTTPRequestParser, HTTPResponseFormatter
from cv_ninja.predictors.input_types import (
    Base64ImagePrediction,
    BatchImagePrediction,
    ImagePrediction,
    Prediction,
    URLImagePrediction,
)
from cv_ninja.predictors.output_formatter import PredictionOutputFormatter
from cv_ninja.predictors.tiling import ImageTiler

__all__ = [
    # Authentication
    "APIKeyAuth",
    "IAMTokenAuth",
    # API Client
    "PredictionAPIClient",
    "FormDataPredictor",
    "BinaryPredictor",
    # Configuration
    "PredictionConfig",
    # Format Converters
    "FormDataFormatConverter",
    "BinaryFormatConverter",
    # Tiling
    "ImageTiler",
    # HTTP Handling
    "HTTPRequestParser",
    "HTTPResponseFormatter",
    # Input Types
    "Prediction",
    "ImagePrediction",
    "Base64ImagePrediction",
    "URLImagePrediction",
    "BatchImagePrediction",
    # Output Formatting
    "PredictionOutputFormatter",
]
