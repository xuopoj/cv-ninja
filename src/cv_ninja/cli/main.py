"""Main CLI entry point for CV Ninja."""

import click

from cv_ninja import __version__
from cv_ninja.cli import commands


@click.group()
@click.version_option(version=__version__, prog_name='cv-ninja')
def cli():
    """CV Ninja - Computer Vision annotation format converter and toolkit.

    A tool for converting between different annotation formats, splitting
    and merging annotations, and processing computer vision datasets.
    """
    pass


@cli.group()
def convert():
    """Convert between annotation formats.

    Supports conversion between various computer vision annotation formats
    including Pascal VOC, Label Studio, LabelMe, and more.
    """
    pass


# Register conversion subcommands
convert.add_command(commands.voc_to_labelstudio)
convert.add_command(commands.labelme_to_labelstudio)


@cli.group()
def predict():
    """Generate predictions using external CV model APIs.

    Send images to trained computer vision models via HTTP APIs
    and convert results to various annotation formats.
    """
    pass


# Register prediction subcommands
predict.add_command(commands.predict_image)
predict.add_command(commands.predict_batch)


# Placeholder for future command groups
@cli.group()
def split():
    """Split annotations and images.

    Tools for splitting large images and their annotations into smaller tiles.
    """
    click.echo("Split commands coming soon!")


@cli.group()
def merge():
    """Merge annotations and images.

    Tools for merging tiled images and annotations back into larger images.
    """
    click.echo("Merge commands coming soon!")


if __name__ == '__main__':
    cli()
