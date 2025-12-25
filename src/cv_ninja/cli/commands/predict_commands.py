"""Prediction commands for CV Ninja CLI."""

import click
import json
from pathlib import Path

from cv_ninja.predictors.base import FormDataPredictor, BinaryPredictor
from cv_ninja.predictors.auth import APIKeyAuth, IAMTokenAuth
from cv_ninja.predictors.config import PredictionConfig
from cv_ninja.predictors.tiling import ImageTiler
from cv_ninja.predictors.input_types import BatchImagePrediction, ImagePrediction
from cv_ninja.predictors.output_formatter import PredictionOutputFormatter
from cv_ninja.utils.exceptions import PredictionError, ValidationError
from PIL import Image


def create_auth_handler(config, api_key_cli, iam_url_cli, username_cli, password_cli, iam_domain_cli=None, iam_project_cli=None):
    """Create appropriate auth handler based on config and CLI options.

    Precedence: CLI options > Profile auth_type > .env file

    Args:
        config: PredictionConfig instance
        api_key_cli: API key from CLI
        iam_url_cli: IAM URL from CLI
        username_cli: Username from CLI
        password_cli: Password from CLI
        iam_domain_cli: IAM domain from CLI
        iam_project_cli: IAM project from CLI

    Returns:
        AuthHandler instance or None

    Raises:
        click.UsageError: If invalid auth options provided
    """
    # Check if profile specifies auth_type
    auth_type = config.get_auth_type()

    # Get values with precedence: CLI > Profile > .env
    api_key = config.get_api_key(api_key_cli)
    iam_url = config.get_iam_url(iam_url_cli)
    username = config.get_username(username_cli)
    password = config.get_password(password_cli)
    iam_domain = config.get_iam_domain(iam_domain_cli)
    iam_project = config.get_iam_project(iam_project_cli)

    # If CLI provides explicit auth, use it
    if api_key_cli or iam_url_cli:
        # CLI auth options override profile auth_type
        if api_key_cli and iam_url_cli:
            raise click.UsageError(
                "Cannot use both --api-key and --iam-url options."
            )

        if api_key_cli:
            return APIKeyAuth(api_key)

        if iam_url_cli:
            if not username or not password:
                raise click.UsageError(
                    "IAM authentication requires username and password. "
                    "Provide --username and --password (or set in .env file)."
                )
            if not iam_domain or not iam_project:
                raise click.UsageError(
                    "IAM authentication requires domain and project. "
                    "Provide --iam-domain and --iam-project (or set in .env file)."
                )
            return IAMTokenAuth(iam_url, username, password, iam_domain, iam_project)

    # Use profile's auth_type if specified
    if auth_type:
        if auth_type == 'api_key':
            if not api_key:
                raise click.UsageError(
                    "Profile specifies api_key auth but PREDICTION_API_KEY not set in .env"
                )
            return APIKeyAuth(api_key)

        elif auth_type == 'iam':
            if not iam_url:
                raise click.UsageError(
                    "Profile specifies iam auth but PREDICTION_IAM_URL not set in .env"
                )
            if not username or not password:
                raise click.UsageError(
                    "IAM authentication requires username and password in .env"
                )
            if not iam_domain or not iam_project:
                raise click.UsageError(
                    "IAM authentication requires domain and project in .env"
                )
            return IAMTokenAuth(iam_url, username, password, iam_domain, iam_project)

        else:
            raise click.UsageError(
                f"Invalid auth_type '{auth_type}' in profile. Use 'api_key' or 'iam'."
            )

    # No profile auth_type - use old logic (error if both are set)
    if api_key and iam_url:
        raise click.UsageError(
            "Cannot use both API key and IAM authentication. "
            "Either set auth_type in profile config, or use only one auth method in .env"
        )

    if api_key:
        return APIKeyAuth(api_key)

    if iam_url:
        if not username or not password:
            raise click.UsageError(
                "IAM authentication requires username and password. "
                "Provide --username and --password (or set in .env file)."
            )
        if not iam_domain or not iam_project:
            raise click.UsageError(
                "IAM authentication requires domain and project. "
                "Provide --iam-domain and --iam-project (or set in .env file)."
            )
        return IAMTokenAuth(iam_url, username, password, iam_domain, iam_project)

    return None


@click.command("image")
@click.argument("image_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "-u",
    "--api-url",
    help="Prediction API endpoint URL (or set PREDICTION_API_URL in .env)",
)
@click.option(
    "-o",
    "--output",
    default="predictions.json",
    type=click.Path(),
    help="Output annotation file (default: predictions.json)",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    default="labelstudio",
    type=click.Choice(["labelstudio", "voc", "coco"]),
    help="Output annotation format (default: labelstudio)",
)
@click.option(
    "--api-key",
    help="API key for Bearer token auth (or set PREDICTION_API_KEY in .env)",
)
@click.option(
    "--iam-url",
    help="IAM service URL for X-Auth-Token auth (or set PREDICTION_IAM_URL in .env)",
)
@click.option(
    "--username",
    help="Username for IAM authentication (or set PREDICTION_USERNAME in .env)",
)
@click.option(
    "--password",
    help="Password for IAM authentication (or set PREDICTION_PASSWORD in .env)",
)
@click.option(
    "--iam-domain",
    help="IAM domain for authentication (or set PREDICTION_IAM_DOMAIN in .env)",
)
@click.option(
    "--iam-project",
    help="IAM project for authentication (or set PREDICTION_IAM_PROJECT in .env)",
)
@click.option(
    "--binary",
    is_flag=True,
    help="Use binary upload mode (raw binary data instead of multipart form)",
)
@click.option(
    "--endpoint",
    default="/upload",
    help="API endpoint path for binary mode (default: /upload)",
)
@click.option(
    "--params",
    help="Query parameters for binary mode as JSON (e.g., '{\"Station_id\": \"123\"}')",
)
@click.option(
    "--tile",
    is_flag=True,
    help="Enable automatic tiling for large images (>1386x1516)",
)
@click.option(
    "--tile-size",
    type=str,
    default="1386x1516",
    help="Tile size in WIDTHxHEIGHT format (default: 1386x1516)",
)
@click.option(
    "--tile-overlap",
    type=int,
    default=32,
    help="Overlap between tiles in pixels (default: 32)",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    help="Path to .env file (default: searches current dir and parents)",
)
@click.option(
    "--profile",
    help="Profile name from YAML config (e.g., prod, test, staging)",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to YAML config file (default: cv-ninja.yaml or endpoints.yaml)",
)
@click.option(
    "--prefix",
    default="",
    help="URL prefix for image paths in Label Studio format (e.g., /data/local-files/?d=)",
)
@click.option(
    "--ls-mode",
    type=click.Choice(["annotations", "predictions"]),
    default="annotations",
    help="Label Studio output mode: 'annotations' or 'predictions' (default: annotations)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def predict_image(
    image_path,
    api_url,
    output,
    output_format,
    api_key,
    iam_url,
    username,
    password,
    iam_domain,
    iam_project,
    binary,
    endpoint,
    params,
    tile,
    tile_size,
    tile_overlap,
    env_file,
    profile,
    config_file,
    prefix,
    ls_mode,
    verbose,
):
    """Predict objects in a single image using external API.

    Credentials can be provided via CLI options or .env file.
    CLI options take precedence over .env values.

    \b
    IMAGE_PATH: Path to image file

    \b
    Examples:
        # Using .env file for credentials
        cv-ninja predict image path/to/image.jpg

        # Override with CLI options
        cv-ninja predict image path/to/image.jpg \\
            -u https://api.example.com/predict \\
            --api-key YOUR_API_KEY

        # IAM authentication from .env
        cv-ninja predict image path/to/image.jpg \\
            --iam-url https://iam.example.com/token \\
            --username user \\
            --password pass

        # Custom .env file location
        cv-ninja predict image path/to/image.jpg \\
            --env-file /path/to/.env
    """
    try:
        # Load configuration (with profile support)
        config = PredictionConfig(env_file, profile=profile, config_file=config_file)

        # Get API URL with precedence: CLI > Profile > .env
        api_url = config.get_api_url(api_url)

        if not api_url:
            raise click.UsageError(
                "API URL is required. Provide --api-url or set PREDICTION_API_URL in .env file."
            )

        if verbose:
            click.echo(f"Predicting objects in: {image_path}")
            click.echo(f"API URL: {api_url}")
            click.echo(f"Output format: {output_format}")

        # Validate input
        prediction = ImagePrediction(
            image_path=image_path,
            output_format=output_format,
        )
        prediction.validate()

        # Create auth handler
        auth_handler = create_auth_handler(config, api_key, iam_url, username, password, iam_domain, iam_project)

        # Determine mode: CLI flag > profile config > default (formdata)
        use_binary = binary or config.get_mode() == 'binary'

        # Get endpoint from profile if not specified
        if use_binary and not endpoint:
            endpoint = config.get_endpoint() or "/upload"

        # Parse tile size
        tile_width, tile_height = 1386, 1516
        if tile_size:
            try:
                parts = tile_size.lower().split('x')
                tile_width = int(parts[0])
                tile_height = int(parts[1])
            except (ValueError, IndexError):
                raise click.UsageError(
                    f"Invalid tile size format: {tile_size}. Use WIDTHxHEIGHT (e.g., 1386x1516)"
                )

        if verbose and tile:
            click.echo(f"Tiling enabled: {tile_width}x{tile_height} with {tile_overlap}px overlap")

        # Create appropriate predictor based on mode
        if use_binary:
            # Parse params if provided
            query_params = {}
            if params:
                try:
                    query_params = json.loads(params)
                except json.JSONDecodeError as e:
                    raise click.UsageError(f"Invalid JSON in --params: {e}")

            if verbose:
                click.echo("Sending request to API...")
                click.echo(f"Using binary upload mode (endpoint: {endpoint})")

            # Create predictor (has built-in converter, returns COCO)
            client = BinaryPredictor(api_url, auth_handler, endpoint=endpoint)

            # Check if tiling is needed
            img = Image.open(image_path)
            tiler = ImageTiler(tile_size=(tile_width, tile_height), overlap=tile_overlap)

            if tile and tiler.needs_tiling(img):
                if verbose:
                    click.echo(f"Image requires tiling: {img.size[0]}x{img.size[1]}")
                # Use tiler to orchestrate prediction (returns COCO)
                result = tiler.predict_tiled(client, img, params=query_params)
            else:
                # Direct prediction (returns COCO)
                result = client.predict_from_file(image_path, params=query_params)
        else:
            if verbose:
                click.echo("Sending request to API...")
                click.echo("Using form-data upload mode")

            # Create predictor (has built-in converter, returns COCO)
            client = FormDataPredictor(api_url, auth_handler)

            # Check if tiling is needed
            img = Image.open(image_path)
            tiler = ImageTiler(tile_size=(tile_width, tile_height), overlap=tile_overlap)

            if tile and tiler.needs_tiling(img):
                if verbose:
                    click.echo(f"Image requires tiling: {img.size[0]}x{img.size[1]}")
                # Use tiler to orchestrate prediction (returns COCO)
                result = tiler.predict_tiled(client, img)
            else:
                # Direct prediction (returns COCO)
                result = client.predict_from_file(image_path)

        # Format output based on requested format
        formatter = PredictionOutputFormatter()

        if output_format == "labelstudio":
            formatted = formatter.to_labelstudio(result, prefix=prefix, output_mode=ls_mode)
            with open(output, "w") as f:
                json.dump(formatted, f, indent=2)

        elif output_format == "voc":
            formatted = formatter.to_voc(result, Path(image_path).name)
            with open(output, "w") as f:
                f.write(formatted)

        elif output_format == "coco":
            formatted = formatter.to_coco(result, image_id=1)
            with open(output, "w") as f:
                json.dump(formatted, f, indent=2)

        click.echo(
            click.style(
                f"✓ Successfully generated predictions for {Path(image_path).name}",
                fg="green",
            )
        )

        if verbose:
            click.echo(f"Output saved to: {output}")
            if result.get("annotations"):
                click.echo(f"Found {len(result['annotations'])} objects")

    except ValidationError as e:
        click.echo(click.style(f"✗ Validation error: {e}", fg="red"), err=True)
        raise click.Abort()
    except PredictionError as e:
        click.echo(click.style(f"✗ Prediction error: {e}", fg="red"), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"✗ Unexpected error: {e}", fg="red"), err=True)
        if verbose:
            raise
        raise click.Abort()


@click.command("batch")
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "-u",
    "--api-url",
    help="Prediction API endpoint URL (or set PREDICTION_API_URL in .env)",
)
@click.option(
    "-o",
    "--output",
    default="predictions.json",
    type=click.Path(),
    help="Output annotation file (default: predictions.json)",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    default="labelstudio",
    type=click.Choice(["labelstudio", "voc", "coco"]),
    help="Output annotation format",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Process images recursively in subdirectories",
)
@click.option(
    "--api-key",
    help="API key for Bearer token auth (or set PREDICTION_API_KEY in .env)",
)
@click.option(
    "--iam-url",
    help="IAM service URL for X-Auth-Token auth (or set PREDICTION_IAM_URL in .env)",
)
@click.option(
    "--username",
    help="Username for IAM authentication (or set PREDICTION_USERNAME in .env)",
)
@click.option(
    "--password",
    help="Password for IAM authentication (or set PREDICTION_PASSWORD in .env)",
)
@click.option(
    "--iam-domain",
    help="IAM domain for authentication (or set PREDICTION_IAM_DOMAIN in .env)",
)
@click.option(
    "--iam-project",
    help="IAM project for authentication (or set PREDICTION_IAM_PROJECT in .env)",
)
@click.option(
    "--binary",
    is_flag=True,
    help="Use binary upload mode (raw binary data instead of multipart form)",
)
@click.option(
    "--endpoint",
    default="/upload",
    help="API endpoint path for binary mode (default: /upload)",
)
@click.option(
    "--params",
    help="Query parameters for binary mode as JSON (e.g., '{\"Station_id\": \"123\"}')",
)
@click.option(
    "--tile",
    is_flag=True,
    help="Enable automatic tiling for large images (>1386x1516)",
)
@click.option(
    "--tile-size",
    type=str,
    default="1386x1516",
    help="Tile size in WIDTHxHEIGHT format (default: 1386x1516)",
)
@click.option(
    "--tile-overlap",
    type=int,
    default=32,
    help="Overlap between tiles in pixels (default: 32)",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    help="Path to .env file (default: searches current dir and parents)",
)
@click.option(
    "--profile",
    help="Profile name from YAML config (e.g., prod, test, staging)",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to YAML config file (default: cv-ninja.yaml or endpoints.yaml)",
)
@click.option(
    "--prefix",
    default="",
    help="URL prefix for image paths in Label Studio format (e.g., /data/local-files/?d=)",
)
@click.option(
    "--ls-mode",
    type=click.Choice(["annotations", "predictions"]),
    default="annotations",
    help="Label Studio output mode: 'annotations' or 'predictions' (default: annotations)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def predict_batch(
    image_dir,
    api_url,
    output,
    output_format,
    recursive,
    api_key,
    iam_url,
    username,
    password,
    iam_domain,
    iam_project,
    binary,
    endpoint,
    params,
    tile,
    tile_size,
    tile_overlap,
    env_file,
    profile,
    config_file,
    prefix,
    ls_mode,
    verbose,
):
    """Predict objects in all images in a directory using external API.

    Credentials can be provided via CLI options or .env file.
    CLI options take precedence over .env values.

    \b
    IMAGE_DIR: Directory containing images

    \b
    Examples:
        # Using .env file for credentials
        cv-ninja predict batch ./images

        # With recursive processing
        cv-ninja predict batch ./images -r

        # Override with CLI options
        cv-ninja predict batch ./images \\
            -u https://api.example.com/predict \\
            --api-key YOUR_API_KEY

        # Custom .env file
        cv-ninja predict batch ./images \\
            --env-file /path/to/.env
    """
    try:
        # Load configuration (with profile support)
        config = PredictionConfig(env_file, profile=profile, config_file=config_file)

        # Get API URL with precedence: CLI > Profile > .env
        api_url = config.get_api_url(api_url)

        if not api_url:
            raise click.UsageError(
                "API URL is required. Provide --api-url or set PREDICTION_API_URL in .env file."
            )

        if verbose:
            click.echo(f"Batch predicting in: {image_dir}")
            click.echo(f"API URL: {api_url}")
            click.echo(f"Output format: {output_format}")
            click.echo(f"Recursive: {recursive}")

        # Validate input
        prediction = BatchImagePrediction(
            image_dir=image_dir,
            output_format=output_format,
            recursive=recursive,
        )
        prediction.validate()

        images = prediction.get_images()
        if verbose:
            click.echo(f"Found {len(images)} images")

        if not images:
            click.echo(
                click.style(f"✗ No images found in {image_dir}", fg="yellow"),
                err=True,
            )
            return

        # Create auth handler
        auth_handler = create_auth_handler(config, api_key, iam_url, username, password, iam_domain, iam_project)

        # Determine mode: CLI flag > profile config > default (formdata)
        use_binary = binary or config.get_mode() == 'binary'

        # Get endpoint from profile if not specified
        if use_binary and not endpoint:
            endpoint = config.get_endpoint() or "/upload"

        # Parse tile size
        tile_width, tile_height = 1386, 1516
        if tile_size:
            try:
                parts = tile_size.lower().split('x')
                tile_width = int(parts[0])
                tile_height = int(parts[1])
            except (ValueError, IndexError):
                raise click.UsageError(
                    f"Invalid tile size format: {tile_size}. Use WIDTHxHEIGHT (e.g., 1386x1516)"
                )

        if verbose and tile:
            click.echo(f"Tiling enabled: {tile_width}x{tile_height} with {tile_overlap}px overlap")

        # Parse params if provided (for binary mode)
        query_params = {}
        if use_binary and params:
            try:
                query_params = json.loads(params)
            except json.JSONDecodeError as e:
                raise click.UsageError(f"Invalid JSON in --params: {e}")

        # Create appropriate predictor based on mode (has built-in converter, returns COCO)
        if use_binary:
            client = BinaryPredictor(api_url, auth_handler, endpoint=endpoint)
        else:
            client = FormDataPredictor(api_url, auth_handler)

        # Create tiler if tiling is enabled
        tiler = ImageTiler(tile_size=(tile_width, tile_height), overlap=tile_overlap) if tile else None

        # Process all images
        all_results = []
        formatter = PredictionOutputFormatter()

        for i, image_path in enumerate(images, 1):
            if verbose:
                click.echo(f"Processing {i}/{len(images)}: {image_path.name}")

            # Load image to check if tiling is needed
            img = Image.open(image_path)

            # Determine if we should use tiling
            if tile and tiler and tiler.needs_tiling(img):
                if verbose:
                    click.echo(f"  Image requires tiling: {img.size[0]}x{img.size[1]}")
                # Use tiler to orchestrate prediction (returns COCO)
                if use_binary:
                    result = tiler.predict_tiled(client, img, params=query_params)
                else:
                    result = tiler.predict_tiled(client, img)
            else:
                # Direct prediction (returns COCO)
                if use_binary:
                    result = client.predict_from_file(str(image_path), params=query_params)
                else:
                    result = client.predict_from_file(str(image_path))

            result["image_name"] = image_path.name

            if output_format == "labelstudio":
                formatted = formatter.to_labelstudio(result, prefix=prefix, output_mode=ls_mode)
                all_results.append(formatted)
            elif output_format == "voc":
                # For VOC, save individual XML files
                xml_output = formatter.to_voc(result, image_path.name)
                xml_path = Path(output).parent / f"{image_path.stem}.xml"
                with open(xml_path, "w") as f:
                    f.write(xml_output)
            elif output_format == "coco":
                formatted = formatter.to_coco(result, image_id=i)
                all_results.extend(formatted)

        # Save results
        if output_format in ["labelstudio", "coco"]:
            with open(output, "w") as f:
                json.dump(all_results, f, indent=2)

        click.echo(
            click.style(f"✓ Successfully processed {len(images)} images", fg="green")
        )

        if verbose:
            click.echo(f"Output saved to: {output}")

    except ValidationError as e:
        click.echo(click.style(f"✗ Validation error: {e}", fg="red"), err=True)
        raise click.Abort()
    except PredictionError as e:
        click.echo(click.style(f"✗ Prediction error: {e}", fg="red"), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"✗ Unexpected error: {e}", fg="red"), err=True)
        if verbose:
            raise
        raise click.Abort()
