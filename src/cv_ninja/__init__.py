"""CV Ninja - Computer Vision annotation format converter and toolkit."""

__version__ = "0.1.0"

from cv_ninja.converters.voc_to_labelstudio import VOCToLabelStudioConverter

__all__ = [
    "__version__",
    "VOCToLabelStudioConverter",
]
