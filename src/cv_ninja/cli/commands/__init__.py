"""CLI commands for CV Ninja."""

from cv_ninja.cli.commands.convert_commands import voc_to_labelstudio
from cv_ninja.cli.commands.predict_commands import predict_batch, predict_image

__all__ = ["voc_to_labelstudio", "predict_image", "predict_batch"]
