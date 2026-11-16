# Pole Vegetation Inspection — Implementation Reference

This document maps every file, directory, and key function that forms the **Pole Vegetation** feature of the GIS-YOLO system.

---

## Directory Layout

```
gis_yolo/
├── app.py                                    # Flask entry-point; registers all route blueprints
├── config.py                                 # Environment / Azure config loader
├── db.py                                     # PostgreSQL connection helpers + schema DDL
│
├── routes/
│   └── pole_vegetation_routes.py             # HTTP endpoints for pole-vegetation analysis
│
├── services/
│   ├── vegetation_context_service.py         # Core image analysis + Azure OpenAI integration
│   ├── pole_vegetation_report_service.py     # Orchestration: build, persist, and retrieve reports
│   ├── distance_estimation_service.py        # Pixel-to-metre distance maths
│   ├── image_crop_service.py                 # Crop tree regions from original photo
│   └── plantnet_service.py                   # PlantNet species identification wrapper
│
├── static/
│   └── script.js                             # Frontend JS (map + report rendering)
│
└── uploads/
    └── pole_vegetation/
        └── <report_id>/                      # Per-report directory (original.jpg, tree crops)
```

---

## File-by-File Reference

### `app.py`
| Lines | Purpose |
|-------|---------|
| 10 | `from routes.pole_vegetation_routes import register_pole_vegetation_routes` |
| 23 | `register_pole_vegetation_routes(app)` — mounts all three endpoints |

---

### `db.py`
| Lines | Purpose |
|-------|---------|
| 113–127 | `CREATE TABLE IF NOT EXISTS pole_vegetation_reports` DDL — columns: `id`, `pole_id`, `latitude`, `longitude`, `capture_date`, `season`, `created_at`, `original_image_path`, `original_uploaded_image_id`, `azure_context_json`, `report_json`, `user_selected_tree_id` |

---

### `routes/pole_vegetation_routes.py` — 56 lines total

| Lines | Endpoint / Function | Description |
|-------|-------------------|-------------|
| 11 | `register_pole_vegetation_routes(app)` | Registration function called in `app.py` |
| 12–30 | `POST /pole-vegetation/analyze` → `pole_vegetation_analyze()` | Accepts a multipart image + optional `lat`, `lon`, `pole_id`, `capture_date`; returns the full report JSON |
| 32–47 | `POST /pole-vegetation/select-tree` → `pole_vegetation_select_tree()` | Re-runs PlantNet species detection for a user-selected tree; requires `report_id` + `user_selected_tree_id` |
| 49–55 | `GET /pole-vegetation/report/<report_id>` → `pole_vegetation_report()` | Fetches a previously saved report from the database |

All endpoints are protected by `@login_required` (lines 13, 33, 50).

---

### `services/pole_vegetation_report_service.py` — 205 lines total

| Lines | Function | Description |
|-------|----------|-------------|
| 12 | `UPLOAD_ROOT = Path('uploads') / 'pole_vegetation'` | Root for all uploaded images |
| 13 | `RISKY_LEVELS = {'medium', 'high', 'critical'}` | Set used to gate species detection |
| 16–17 | `_now_iso()` | Returns current datetime as ISO string |
| 20–27 | `_ensure_plantnet_shape(tree)` | Populates default `plantnet_result` dict on a tree if absent |
| 30–58 | `_report_table_upsert(...)` | INSERT … ON CONFLICT DO UPDATE into `pole_vegetation_reports`; writes or refreshes a row atomically |
| 61–71 | `get_report(report_id)` | SELECT by primary key; decodes `report_json`; returns dict or `None` |
| 74–87 | `_save_report(report_id, report)` | Thin wrapper that extracts fields from the report dict and calls `_report_table_upsert` |
| 90–102 | `_apply_species_detection(tree, original_image_path, report_dir)` | Crops the tree region → calls PlantNet → writes result back; upgrades `recommended_action` to `capture_side_angle` when confidence < 0.35 |
| 105–176 | `analyze_pole_vegetation(image_storage, ...)` | **Main entry point** — orchestrates: save image → infer season → call Azure context analysis → species detection for risky trees → assemble + persist full report dict |
| 179–204 | `rerun_species_detection_for_selected_tree(report_id, user_selected_tree_id)` | Loads an existing report, finds the chosen tree by ID, re-runs `_apply_species_detection`, persists update |

---

### `services/vegetation_context_service.py` — 662 lines total

#### Utility / Geometry Helpers
| Lines | Function | Description |
|-------|----------|-------------|
| 26–58 | `infer_season(capture_date, latitude)` | Determines season from month + hemisphere |
| 61–66 | `_decode_image(image_bytes)` | Decodes raw bytes to OpenCV numpy array via `cv2.imdecode` |
| 69–79 | `_bbox_from_detection(det)` | Extracts `[x1,y1,x2,y2]` from a YOLO detection dict (supports `bbox` or `polygon`) |
| 82–89 | `_normalize_bbox(box, width, height)` | Converts pixel box to `{x_min, y_min, x_max, y_max}` in [0, 1] |
| 92–99 | `_clip_box(box, width, height)` | Clamps pixel coords to image bounds |
| 102–124 | `_normalize_line(line, width, height)` | Normalises a wire line dict; auto-detects pixel vs normalised coords |

#### Wire / Vegetation Detection
| Lines | Function | Description |
|-------|----------|-------------|
| 127–144 | `_wire_lines_from_detections(detections, width, height)` | Filters CONDUCTOR / WIRE detections and converts bounding boxes to line segments |
| 246–268 | `_merge_boxes(boxes, merge_gap=40)` | Iterative bbox merge with inflated proximity test |
| 271–310 | `_detect_vegetation_regions(image, detections)` | HSV colour segmentation (green + brown ranges) + connected components; merges regions; falls back to full frame if nothing detected |

#### Geometry Risk Scoring
| Lines | Function | Description |
|-------|----------|-------------|
| 230–237 | `_box_intersection(a, b)` | Returns intersection area of two boxes |
| 240–243 | `_box_distance(a, b)` | Min edge-to-edge Euclidean distance |
| 338–348 | `_relative_position(tree_box, pole_box)` | Returns `left`, `right`, or `front` from centre-x comparison |
| 351–366 | `_risk_from_geometry(tree_box, pole_boxes, wire_boxes, image_diag)` | 5-level risk: touching → critical, ≤4 % diag → high, ≤10 % → medium, ≤18 % → low |
| 369–376 | `_recommended_action(risk_level)` | Maps risk level to action string |

#### Measurement Bundle
| Lines | Function | Description |
|-------|----------|-------------|
| 147–227 | `_build_measurement_bundle(trees, pole_box, wire_lines, width, height, ...)` | Computes `meters_per_pixel` from assumed 9 m pole; calls `build_distance_metrics_for_tree` for each tree; upgrades `risk_level` from geometry; builds `tree_to_tree_distances` |

#### Context Analysis (Azure + Fallback)
| Lines | Function | Description |
|-------|----------|-------------|
| 313–325 | `_detected_object_summary(detections)` | Separates pole and wire bounding boxes from YOLO detections |
| 328–335 | `_recompute_overall_risk(trees)` | Picks highest risk across all trees |
| 379–386 | `_empty_plantnet_result()` | Default PlantNet result shape |
| 389–400 | `_build_summary(trees)` | Human-readable summary sentence for the report |
| 403–438 | `_normalize_tree_item(raw, width, height, idx)` | Sanitises and clamps an Azure-returned tree dict |
| 441–519 | `_fallback_context_analysis(image_bytes, ...)` | Pure CV fallback: YOLO detect → HSV vegetation segmentation → geometry risk scoring; used when Azure is unavailable |
| 522–661 | `analyze_pole_context_with_azure(image_bytes, ...)` | **Primary analysis function** — runs fallback first, then attempts Azure OpenAI vision calls (up to 3 token budgets per config); merges and normalises Azure response; returns full context dict |

---

### `services/distance_estimation_service.py` — 245 lines total

| Lines | Function | Description |
|-------|----------|-------------|
| 4–16 | `normalized_bbox_to_px(bbox, w, h)` | Converts normalised bbox dict to pixel coords |
| 19–29 | `normalized_line_to_px(line, w, h)` | Same for wire line dicts |
| 32–35 | `calculate_bbox_height_px(bbox_px)` | Returns `y2 - y1` |
| 38–49 | `point_to_line_segment_distance_px(point, line)` | Perpendicular distance from a point to a line segment + nearest projected point |
| 52–53 | `_point_inside_bbox(point, bbox_px)` | Boolean containment check |
| 56–81 | `_segments_intersect(a1, a2, b1, b2)` | General 2D segment intersection test using orientation + collinearity |
| 84–101 | `_line_touches_bbox(line_px, bbox_px)` | True if wire line endpoints are inside the bbox or any edge intersects |
| 104–137 | `bbox_to_line_distance_px(bbox_px, line_px)` | Returns `{distance_px, is_touching, nearest_point_on_bbox, nearest_point_on_line}` from 8 candidate bbox points |
| 140–145 | `bbox_to_bbox_distance_px(bbox1_px, bbox2_px)` | Edge-to-edge distance between two bboxes |
| 148–163 | `estimate_distance_using_pole_height(pole_height_px, distance_px, ...)` | `distance_m = distance_px × (9.0 / pole_height_px)` |
| 166–177 | `classify_clearance_risk(distance_m, is_touching)` | Clearance thresholds: <0.5 m → high, <1.5 m → medium, <3.0 m → low |
| 180–191 | `_clearance_label(distance_m, is_touching)` | Human label: touching / very_close / close / moderate / far |
| 194–244 | `build_distance_metrics_for_tree(tree_bbox, wire_lines, pole_bbox, w, h, ...)` | **Main metric builder** — finds nearest wire, estimates clearance in metres, computes pole distance, returns full metrics dict with `recomputed_risk_level` |

---

### `services/image_crop_service.py` — 57 lines total

| Lines | Function | Description |
|-------|----------|-------------|
| 6–7 | `_clamp01(value)` | Clamps float to [0, 1] |
| 10–31 | `validate_bbox(bbox)` | Validates normalised bbox dict; raises `ValueError` on invalid shape |
| 34–56 | `crop_image_by_normalized_bbox(image_path, bbox, output_path, padding=0.08)` | Opens image with PIL, expands bbox by 8 % padding, crops and saves as JPEG at quality 95 |

---

### `services/plantnet_service.py` — 20 lines total

| Lines | Function | Description |
|-------|----------|-------------|
| 6–19 | `identify_species(image_path)` | Reads image bytes from disk → calls `identify_species_with_plantnet` from `tree_routes` → returns `{best_match, scientific_name, common_name, confidence, top_predictions[:5]}` |

---

## API Endpoints Summary

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/pole-vegetation/analyze` | ✅ | Upload image; returns full vegetation report |
| `POST` | `/pole-vegetation/select-tree` | ✅ | Re-run species detection for a chosen tree |
| `GET` | `/pole-vegetation/report/<report_id>` | ✅ | Fetch a saved report by ID |

### `POST /pole-vegetation/analyze` — Request Form Fields

| Field | Required | Description |
|-------|----------|-------------|
| `image` | ✅ | JPEG/PNG image file |
| `lat` / `latitude` | ➖ | GPS latitude (float) |
| `lon` / `longitude` | ➖ | GPS longitude (float) |
| `pole_id` | ➖ | External pole identifier string |
| `capture_date` | ➖ | ISO-8601 date/time string |

---

## Data Flow

```
POST /pole-vegetation/analyze
  │
  ├─ pole_vegetation_routes.py (L19-25)
  │   └─ analyze_pole_vegetation()
  │
  ├─ pole_vegetation_report_service.py (L105-176)
  │   ├─ infer_season()                           # vegetation_context_service.py L26
  │   ├─ save_uploaded_image()                    # db.py
  │   ├─ analyze_pole_context_with_azure()        # vegetation_context_service.py L522
  │   │   ├─ _fallback_context_analysis()         # L441 — CV fallback always runs first
  │   │   │   ├─ process_image_file()             # image_processing.py (YOLO inference)
  │   │   │   ├─ _detect_vegetation_regions()     # L271 — HSV segmentation
  │   │   │   ├─ _risk_from_geometry()            # L351 — pixel-distance risk
  │   │   │   └─ _build_measurement_bundle()      # L147 — pole-scale distance
  │   │   └─ Azure OpenAI vision call             # L592-649 (3 token-budget retries)
  │   │       └─ _normalize_tree_item()           # L403
  │   │       └─ _build_measurement_bundle()      # L618
  │   ├─ _apply_species_detection()               # L90 — for risky trees
  │   │   ├─ crop_image_by_normalized_bbox()      # image_crop_service.py L34
  │   │   └─ identify_species()                   # plantnet_service.py L6
  │   └─ _save_report()                           # L74 → _report_table_upsert() L30
  │
  └─ JSON report returned to caller
```

---

## Database Schema

```sql
-- db.py L113-127
CREATE TABLE IF NOT EXISTS pole_vegetation_reports (
    id                         TEXT PRIMARY KEY,
    pole_id                    TEXT,
    latitude                   DOUBLE PRECISION,
    longitude                  DOUBLE PRECISION,
    capture_date               TEXT,
    season                     TEXT,
    created_at                 TEXT NOT NULL,
    original_image_path        TEXT,
    original_uploaded_image_id TEXT REFERENCES uploaded_images(id) ON DELETE SET NULL,
    azure_context_json         TEXT,        -- raw Azure response
    report_json                TEXT NOT NULL,   -- full serialised report dict
    user_selected_tree_id      TEXT
);
```

---

## Report JSON Structure

```jsonc
{
  "report_id": "<uuid>",
  "pole_id": "",
  "input": { "latitude": 0.0, "longitude": 0.0, "capture_date": "...", "season": "..." },
  "measurement_config": { "assumed_pole_height_m": 9.0, "measurement_type": "single_image_estimate" },
  "image_quality": { "is_usable": true, "issues": [], "recommendation": "" },
  "detected_objects": { "pole_visible": true, "wire_visible": true, "vegetation_count": 2 },
  "pole_measurement": {
    "pole_bbox": { "x_min": 0.4, "y_min": 0.05, "x_max": 0.6, "y_max": 0.95 },
    "pole_height_px": 320.0,
    "meters_per_pixel": 0.028,
    "can_estimate_distance": true,
    "reason": ""
  },
  "wire_lines": [ { "x1": 0.0, "y1": 0.3, "x2": 1.0, "y2": 0.3 } ],
  "trees": [
    {
      "tree_id": "tree_1",
      "position": "left",
      "bbox_normalized": { "x_min": 0.1, "y_min": 0.2, "x_max": 0.35, "y_max": 0.9 },
      "risk_level": "high",
      "risk_reason": "...",
      "is_affecting_pole": true,
      "species_detection_required": true,
      "plantnet_result": {
        "best_match": "Eucalyptus globulus",
        "scientific_name": "Eucalyptus globulus",
        "common_name": "Blue Gum",
        "confidence": 0.72,
        "top_predictions": [...]
      },
      "recommended_action": "trim_required",
      "distance_metrics": {
        "pole_height_assumed_m": 9.0,
        "meters_per_pixel": 0.028,
        "nearest_wire_distance_px": 12.5,
        "estimated_wire_clearance_m": 0.35,
        "clearance_label": "very_close",
        "distance_confidence": "medium",
        "is_touching_wire": false,
        "recomputed_risk_level": "high"
      }
    }
  ],
  "tree_to_tree_distances": [...],
  "overall_risk": "high",
  "summary": "1 vegetation region(s) require follow-up. Nearest estimated wire clearance is about 0.35 m.",
  "ask_extra_photo": false,
  "extra_photo_instruction": "",
  "safety_instructions": [
    "Do not touch wires, poles, or tree branches.",
    "Do not stand under wires.",
    "Do not climb the pole or the tree.",
    "Capture photos from a safe distance only."
  ],
  "warnings": [],
  "token_usage": { "provider": "azure_openai", "status": "completed", "total_tokens": 1200 }
}
```

---

## Risk Levels

| Level | Pixel distance (of image diagonal) | Wire clearance | Action |
|-------|------------------------------------|----------------|--------|
| `critical` | Touching | 0 m | `urgent_action` |
| `high` | ≤ 4 % | < 0.5 m | `trim_required` |
| `medium` | ≤ 10 % | < 1.5 m | `supervisor_review` |
| `low` | ≤ 18 % | < 3.0 m | `monitor` |
| `no_risk` | > 18 % | ≥ 3.0 m | `no_action` |

---

## Key Constants & Assumptions

| Constant | Value | Location |
|----------|-------|----------|
| Assumed pole height | 9.0 m | `vegetation_context_service.py` L151; `distance_estimation_service.py` L162 |
| Upload root | `uploads/pole_vegetation/` | `pole_vegetation_report_service.py` L12 |
| Risky risk levels | `{'medium', 'high', 'critical'}` | `pole_vegetation_report_service.py` L13 |
| PlantNet confidence threshold | 0.35 | `pole_vegetation_report_service.py` L100 |
| Image crop padding | 8 % | `image_crop_service.py` L34 |
| JPEG crop quality | 95 | `image_crop_service.py` L54 |
| Merge gap for vegetation boxes | 40 px (or 2 % of width) | `vegetation_context_service.py` L246, 304 |
| Min vegetation area | max(1500 px², 1 % of frame) | `vegetation_context_service.py` L283 |
