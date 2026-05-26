"""Unit tests for VIA→YOLO conversion and KYC validation rules."""
import json
from pathlib import Path

import pytest
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES
from src.data_prep import (
    convert_via_json,
    extract_label,
    polygon_to_yolo_bbox,
    split_dataset,
    via_region_to_polygon,
    write_data_yaml,
)
from src.inference import validate_detection_quality


# ────────────────────────────────────────────────────────────────────
# polygon_to_yolo_bbox
# ────────────────────────────────────────────────────────────────────

def test_polygon_to_yolo_bbox_axis_aligned():
    """Square at center of 200x200 image → cx=0.5, cy=0.5, w=0.5, h=0.5."""
    xs = [50, 150, 150, 50]
    ys = [50, 50, 150, 150]
    cx, cy, w, h = polygon_to_yolo_bbox(xs, ys, 200, 200)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)


def test_polygon_to_yolo_bbox_clamped():
    """Out-of-bounds polygon → values clamped to [0,1]."""
    xs = [-10, 50, 50, -10]
    ys = [-10, -10, 50, 50]
    cx, cy, w, h = polygon_to_yolo_bbox(xs, ys, 100, 100)
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


def test_polygon_to_yolo_bbox_full_image():
    """Polygon covering entire image → cx=0.5, cy=0.5, w=1.0, h=1.0."""
    xs = [0, 640, 640, 0]
    ys = [0, 0, 480, 480]
    cx, cy, w, h = polygon_to_yolo_bbox(xs, ys, 640, 480)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(1.0)


def test_polygon_to_yolo_bbox_top_left_corner():
    """Small bbox at top-left corner."""
    xs = [0, 10]
    ys = [0, 10]
    cx, cy, w, h = polygon_to_yolo_bbox(xs, ys, 100, 100)
    assert cx == pytest.approx(0.05)
    assert cy == pytest.approx(0.05)
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.10)


# ────────────────────────────────────────────────────────────────────
# extract_label
# ────────────────────────────────────────────────────────────────────

def test_extract_label_field_name():
    assert extract_label({"field_name": "Photo"}) == "photo"


def test_extract_label_class_key():
    assert extract_label({"class": "MRZ"}) == "mrz"


def test_extract_label_type_key():
    assert extract_label({"type": "Surname"}) == "surname"


def test_extract_label_strips_whitespace():
    assert extract_label({"field_name": "  date_of_birth  "}) == "date_of_birth"


def test_extract_label_no_known_key():
    assert extract_label({"unknown_key": "value"}) is None


def test_extract_label_empty_dict():
    assert extract_label({}) is None


# ────────────────────────────────────────────────────────────────────
# via_region_to_polygon
# ────────────────────────────────────────────────────────────────────

def test_via_polygon_shape():
    region = {"shape_attributes": {
        "name": "polygon",
        "all_points_x": [10, 20, 20, 10],
        "all_points_y": [10, 10, 20, 20],
    }}
    px, py = via_region_to_polygon(region)
    assert px == [10, 20, 20, 10]
    assert py == [10, 10, 20, 20]


def test_via_rect_shape():
    region = {"shape_attributes": {
        "name": "rect", "x": 5, "y": 10, "width": 30, "height": 40,
    }}
    px, py = via_region_to_polygon(region)
    assert px == [5, 35, 35, 5]
    assert py == [10, 10, 50, 50]


def test_via_polyline_shape():
    region = {"shape_attributes": {
        "name": "polyline",
        "all_points_x": [0, 100],
        "all_points_y": [0, 100],
    }}
    px, py = via_region_to_polygon(region)
    assert px == [0, 100]


def test_via_unsupported_shape_returns_none():
    region = {"shape_attributes": {"name": "circle", "cx": 10, "cy": 10, "r": 5}}
    assert via_region_to_polygon(region) is None


def test_via_missing_shape_attributes():
    region = {}
    assert via_region_to_polygon(region) is None


# ────────────────────────────────────────────────────────────────────
# convert_via_json end-to-end (with a tiny synthetic fixture)
# ────────────────────────────────────────────────────────────────────

def _make_via_fixture(tmp_path, regions, filename="doc1.jpg", img_size=(100, 100)):
    """Helper to create a VIA JSON + dummy image for testing."""
    img_path = tmp_path / filename
    Image.new("RGB", img_size, "white").save(img_path)
    via_data = {
        "_via_img_metadata": {
            f"{filename}_999": {
                "filename": filename,
                "regions": regions,
            }
        }
    }
    via_path = tmp_path / "ann.json"
    via_path.write_text(json.dumps(via_data))
    return via_path


def test_convert_via_json_writes_labels(tmp_path):
    via_path = _make_via_fixture(tmp_path, regions=[
        {
            "shape_attributes": {
                "name": "rect",
                "x": 10, "y": 10, "width": 30, "height": 30,
            },
            "region_attributes": {"field_name": "photo"},
        },
        {
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": [50, 90, 90, 50],
                "all_points_y": [60, 60, 80, 80],
            },
            "region_attributes": {"field_name": "date_of_expiry"},
        },
    ])

    out_imgs = tmp_path / "out_imgs"
    out_lbls = tmp_path / "out_lbls"
    n_ok, n_skip, n_boxes = convert_via_json(via_path, tmp_path, out_imgs, out_lbls)

    assert n_ok == 1
    assert n_skip == 0
    assert n_boxes == 2

    lbl_files = list(out_lbls.glob("*.txt"))
    assert len(lbl_files) == 1
    lines = lbl_files[0].read_text().strip().split("\n")
    assert len(lines) == 2
    cls_ids = sorted(int(line.split()[0]) for line in lines)
    assert cls_ids == [0, 4]


def test_convert_via_json_skips_unknown_labels(tmp_path):
    """Regions with labels not in CLASS_MAPPING should be ignored."""
    via_path = _make_via_fixture(tmp_path, regions=[
        {
            "shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 50, "height": 50},
            "region_attributes": {"field_name": "signature"},
        },
    ])
    out_imgs = tmp_path / "out_imgs"
    out_lbls = tmp_path / "out_lbls"
    n_ok, n_skip, n_boxes = convert_via_json(via_path, tmp_path, out_imgs, out_lbls)

    assert n_ok == 0
    assert n_skip == 1
    assert n_boxes == 0


def test_convert_via_json_skips_missing_image(tmp_path):
    """If the image file doesn't exist, skip it."""
    via_data = {
        "_via_img_metadata": {
            "ghost.jpg_1": {
                "filename": "ghost.jpg",
                "regions": [
                    {
                        "shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 10, "height": 10},
                        "region_attributes": {"field_name": "photo"},
                    },
                ],
            }
        }
    }
    via_path = tmp_path / "ann.json"
    via_path.write_text(json.dumps(via_data))

    out_imgs = tmp_path / "out_imgs"
    out_lbls = tmp_path / "out_lbls"
    n_ok, n_skip, n_boxes = convert_via_json(via_path, tmp_path, out_imgs, out_lbls)

    assert n_ok == 0
    assert n_skip == 1


def test_convert_via_json_label_values_normalized(tmp_path):
    """Check that bbox values are in [0,1] range."""
    via_path = _make_via_fixture(tmp_path, regions=[
        {
            "shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 30, "height": 40},
            "region_attributes": {"field_name": "mrz"},
        },
    ])
    out_imgs = tmp_path / "out_imgs"
    out_lbls = tmp_path / "out_lbls"
    convert_via_json(via_path, tmp_path, out_imgs, out_lbls)

    lbl_file = list(out_lbls.glob("*.txt"))[0]
    parts = lbl_file.read_text().strip().split()
    assert int(parts[0]) == 1
    for val in parts[1:]:
        assert 0.0 <= float(val) <= 1.0


# ────────────────────────────────────────────────────────────────────
# split_dataset
# ────────────────────────────────────────────────────────────────────

def test_split_dataset_ratios(tmp_path):
    """Verify 70/20/10 split on 10 images."""
    imgs_dir = tmp_path / "imgs"
    lbls_dir = tmp_path / "lbls"
    imgs_dir.mkdir()
    lbls_dir.mkdir()

    for i in range(10):
        Image.new("RGB", (50, 50), "red").save(imgs_dir / f"img_{i}.jpg")
        (lbls_dir / f"img_{i}.txt").write_text(f"0 0.5 0.5 0.3 0.3")

    yolo_root = tmp_path / "yolo"
    counts = split_dataset(imgs_dir, lbls_dir, yolo_root)

    assert counts["train"] == 7
    assert counts["val"] == 2
    assert counts["test"] == 1
    assert sum(counts.values()) == 10

    for split in ("train", "val", "test"):
        n_imgs = len(list((yolo_root / "images" / split).iterdir()))
        n_lbls = len(list((yolo_root / "labels" / split).iterdir()))
        assert n_imgs == counts[split]
        assert n_lbls == counts[split]


# ────────────────────────────────────────────────────────────────────
# write_data_yaml
# ────────────────────────────────────────────────────────────────────

def test_write_data_yaml(tmp_path):
    yaml_path = write_data_yaml(tmp_path)
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert f"nc: {NUM_CLASSES}" in content
    for name in CLASS_NAMES:
        assert name in content
    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "test: images/test" in content


# ────────────────────────────────────────────────────────────────────
# config sanity
# ────────────────────────────────────────────────────────────────────

def test_config_class_count():
    assert NUM_CLASSES == 6
    assert len(CLASS_NAMES) == NUM_CLASSES


# ────────────────────────────────────────────────────────────────────
# KYC validation rules
# ────────────────────────────────────────────────────────────────────

def test_validate_rejects_when_no_photo():
    result = {
        "detections": {
            "expiry_date": {"bbox": [0, 0, 10, 10], "conf": 0.9, "crop_path": ""},
        },
        "missing_fields": ["photo", "mrz", "name", "birth_date", "document_number"],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "REJECTED"
    assert any("photo" in w.lower() for w in quality["warnings"])


def test_validate_review_on_low_photo_conf():
    result = {
        "detections": {
            "photo": {"bbox": [0, 0, 10, 10], "conf": 0.55, "crop_path": ""},
            "expiry_date": {"bbox": [0, 0, 10, 10], "conf": 0.92, "crop_path": ""},
        },
        "missing_fields": [],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "REVIEW"


def test_validate_ok_on_clean_detection():
    result = {
        "detections": {
            "photo": {"bbox": [0, 0, 10, 10], "conf": 0.95, "crop_path": ""},
            "mrz": {"bbox": [0, 0, 10, 10], "conf": 0.93, "crop_path": ""},
            "name": {"bbox": [0, 0, 10, 10], "conf": 0.91, "crop_path": ""},
            "birth_date": {"bbox": [0, 0, 10, 10], "conf": 0.88, "crop_path": ""},
            "expiry_date": {"bbox": [0, 0, 10, 10], "conf": 0.92, "crop_path": ""},
            "document_number": {"bbox": [0, 0, 10, 10], "conf": 0.90, "crop_path": ""},
        },
        "missing_fields": [],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "OK"
    assert quality["warnings"] == []


def test_validate_rejects_empty_detections():
    result = {"detections": {}, "missing_fields": []}
    quality = validate_detection_quality(result)
    assert quality["status"] == "REJECTED"


def test_validate_review_on_many_missing_fields():
    result = {
        "detections": {
            "photo": {"bbox": [0, 0, 10, 10], "conf": 0.95, "crop_path": ""},
        },
        "missing_fields": ["mrz", "name", "birth_date", "expiry_date"],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "REVIEW"
    assert any("missing" in w.lower() for w in quality["warnings"])


def test_validate_review_on_low_mean_confidence():
    result = {
        "detections": {
            "photo": {"bbox": [0, 0, 10, 10], "conf": 0.75, "crop_path": ""},
            "mrz": {"bbox": [0, 0, 10, 10], "conf": 0.40, "crop_path": ""},
            "name": {"bbox": [0, 0, 10, 10], "conf": 0.50, "crop_path": ""},
        },
        "missing_fields": [],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "REVIEW"
    assert any("mean confidence" in w.lower() for w in quality["warnings"])


def test_validate_rejected_takes_priority_over_review():
    """No photo = REJECTED, even if mean conf is also low."""
    result = {
        "detections": {
            "mrz": {"bbox": [0, 0, 10, 10], "conf": 0.40, "crop_path": ""},
        },
        "missing_fields": ["photo", "name", "birth_date", "expiry_date", "document_number"],
    }
    quality = validate_detection_quality(result)
    assert quality["status"] == "REJECTED"
