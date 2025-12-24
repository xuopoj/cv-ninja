"""Abstract base class for all annotation format converters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple


class BaseConverter(ABC):
    """Abstract base class for annotation format converters.

    All converters should inherit from this class and implement
    the required abstract methods to ensure consistent interface.
    """

    @abstractmethod
    def convert(self, input_path: str, output_path: str, **kwargs) -> None:
        """Convert annotations from source format to target format.

        Args:
            input_path: Path to input file or directory
            output_path: Path to output file or directory
            **kwargs: Additional converter-specific options

        Raises:
            ConversionError: If conversion fails
        """
        pass

    @abstractmethod
    def validate_input(self, input_path: str) -> bool:
        """Validate that input is in expected format.

        Args:
            input_path: Path to input file or directory

        Returns:
            True if input is valid

        Raises:
            ValidationError: If input validation fails
        """
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> Tuple[str, str]:
        """Return tuple of (source_format, target_format).

        Returns:
            Tuple containing source and target format names

        Example:
            ('pascal_voc', 'labelstudio')
        """
        pass
