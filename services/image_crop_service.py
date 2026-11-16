from pathlib import Path

from PIL import Image


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def validate_bbox(bbox):
    if not isinstance(bbox, dict):
        raise ValueError("Bounding box must be a dictionary")

    keys = ("x_min", "y_min", "x_max", "y_max")
    if not all(key in bbox for key in keys):
        raise ValueError("Bounding box is missing one or more required keys")

    x_min = _clamp01(bbox["x_min"])
    y_min = _clamp01(bbox["y_min"])
    x_max = _clamp01(bbox["x_max"])
    y_max = _clamp01(bbox["y_max"])

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Bounding box coordinates are invalid")

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def crop_image_by_normalized_bbox(image_path, bbox, output_path, padding=0.08):
    bbox = validate_bbox(bbox)
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size
        pad_x = (bbox["x_max"] - bbox["x_min"]) * padding
        pad_y = (bbox["y_max"] - bbox["y_min"]) * padding

        x1 = int(max(0, (bbox["x_min"] - pad_x) * width))
        y1 = int(max(0, (bbox["y_min"] - pad_y) * height))
        x2 = int(min(width, (bbox["x_max"] + pad_x) * width))
        y2 = int(min(height, (bbox["y_max"] + pad_y) * height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Expanded crop box is empty")

        crop = img.crop((x1, y1, x2, y2))
        crop.save(output_path, format="JPEG", quality=95)

    return str(output_path)
