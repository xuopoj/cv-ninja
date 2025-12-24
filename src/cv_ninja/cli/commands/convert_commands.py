"""Conversion commands for CV Ninja CLI."""

import click

from cv_ninja.converters.voc_to_labelstudio import VOCToLabelStudioConverter
from cv_ninja.utils.exceptions import ConversionError, ValidationError


@click.command('voc-to-labelstudio')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    '-o', '--output',
    default='labelstudio.json',
    type=click.Path(),
    help='Output JSON file path (default: labelstudio.json)'
)
@click.option(
    '-p', '--prefix',
    default='',
    help='URL prefix for image paths (e.g., /data/images/)'
)
@click.option(
    '-i', '--image-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Directory containing image files. If provided, dimensions will be read from actual images instead of XML.'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Enable verbose output'
)
def voc_to_labelstudio(input_dir, output, prefix, image_dir, verbose):
    """Convert Pascal VOC annotations to Label Studio JSON format.

    Converts all XML annotation files in INPUT_DIR from Pascal VOC format
    to Label Studio's JSON format with bounding box annotations.

    \b
    INPUT_DIR: Directory containing Pascal VOC XML annotation files

    \b
    Examples:
        # Basic conversion (reads dimensions from XML)
        cv-ninja convert voc-to-labelstudio dataset/voc

        # Read dimensions from actual image files
        cv-ninja convert voc-to-labelstudio dataset/voc -i dataset/images

        # Specify output file
        cv-ninja convert voc-to-labelstudio dataset/voc -o output.json

        # Add URL prefix for images
        cv-ninja convert voc-to-labelstudio dataset/voc \\
            -o output.json \\
            -p "/data/local-files/?d=images/"

        # Complete example with all options
        cv-ninja convert voc-to-labelstudio dataset/voc \\
            -i dataset/images \\
            -o output.json \\
            -p "/data/images/" \\
            -v
    """
    try:
        if verbose:
            click.echo(f"Converting VOC annotations from: {input_dir}")
            click.echo(f"Output file: {output}")
            if prefix:
                click.echo(f"Image URL prefix: {prefix}")
            if image_dir:
                click.echo(f"Reading dimensions from images in: {image_dir}")
            else:
                click.echo(f"Reading dimensions from XML files")

        converter = VOCToLabelStudioConverter(prefix=prefix, image_dir=image_dir)
        converter.convert(input_dir, output)

        # Count tasks in output
        import json
        from pathlib import Path

        if Path(output).exists():
            with open(output) as f:
                tasks = json.load(f)
                task_count = len(tasks)
        else:
            task_count = 0

        click.echo(
            click.style(
                f"✓ Successfully converted {task_count} tasks to {output}",
                fg='green'
            )
        )

        if verbose:
            click.echo(f"Conversion completed!")

    except ValidationError as e:
        click.echo(
            click.style(f"✗ Validation error: {e}", fg='red'),
            err=True
        )
        raise click.Abort()

    except ConversionError as e:
        click.echo(
            click.style(f"✗ Conversion error: {e}", fg='red'),
            err=True
        )
        raise click.Abort()

    except Exception as e:
        click.echo(
            click.style(f"✗ Unexpected error: {e}", fg='red'),
            err=True
        )
        if verbose:
            raise
        raise click.Abort()
