from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
    "BMP",
    "WEBP",
}

MIN_IMAGE_SIZE = 32
MAX_IMAGE_PIXELS = 25_000_000


def validate_image(image_path):

    image_path = Path(image_path)

    # Check whether the file exists.
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"The provided path is not a file: {image_path}"
        )

    try:
        with Image.open(image_path) as image:

            # Check the file format.
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {image.format}"
                )

            width, height = image.size

            # Check image dimensions.
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                raise ValueError(
                    f"Image is too small: {width}x{height}"
                )

            # Prevent excessively large images from consuming
            # too much memory during processing.
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image is too large: {width}x{height}"
                )

            # Verify that the image data can be decoded.
            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(
            f"Invalid or corrupted image: {image_path}"
        ) from exc

    return True