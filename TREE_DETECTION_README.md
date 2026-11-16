# Tree Detection — Implementation Reference

This package contains every file needed to implement the **Tree Detection** feature of the GIS-YOLO system, including PlantNet species identification and Azure OpenAI LLM analysis.

---

## What Is In the Zip

```
tree_detection_implementation.zip
├── app.py                    # Flask entry-point — route registration
├── config.py                 # Thresholds, label routing, env loader
├── db.py                     # PostgreSQL helper + schema DDL
├── image_processing.py       # YOLO pipeline runner (detections)
├── detection_processing.py   # Conductor / hardware detection post-processing
├── routes/
│   ├── __init__.py
│   ├── auth_routes.py        # login_required decorator used by tree route
│   └── tree_routes.py        # /api/tree_detect endpoint + all helpers
└── requirements.txt          # Full Python dependency list
```

> **Models not included** (too large for zip):
> - `models/pole_best6.pt` — pole detector
> - `models/insulator_new.pt` — insulator detector
> - `models/channel_12class_v2.pt` — crossarm / structural detector
> - `models/cable_best.pt` — cable segmentation model
> - `models/shed_model.pt` — shed detector
>
> Place all `.pt` files in a `models/` directory at project root before running.

---

## Directory Layout (Full Project)

```
gis_yolo/
├── app.py                        # Flask entry-point
├── config.py                     # Runtime config
├── db.py                         # PostgreSQL / DB helpers
├── image_processing.py           # YOLO pipeline runner
├── detection_processing.py       # Hardware detection post-processing
├── pipeline.py                   # InfrastructurePipeline (multi-model orchestration)
├── requirements.txt
│
├── models/
│   ├── pole_best6.pt             # Pole detection model
│   ├── insulator_new.pt          # Insulator detection model
│   ├── channel_12class_v2.pt     # Crossarm / structural model
│   ├── cable_best.pt             # Cable segmentation model
│   └── shed_model.pt             # Shed detection model
│
├── routes/
│   ├── __init__.py
│   ├── auth_routes.py            # Session auth + login_required
│   └── tree_routes.py            # Tree detect endpoint + PlantNet + Azure LLM
│
└── uploads/                      # Runtime image storage
```

---

## File-by-File Reference

### `app.py` — 34 lines

| Lines | Purpose |
|-------|---------|
| 11 | `from routes.tree_routes import register_tree_routes` |
| 22 | `register_tree_routes(app)` — mounts `/api/tree_detect` |

---

### `config.py` — 75 lines

| Lines | Symbol | Description |
|-------|--------|-------------|
| 8 | `DETECTION_CONF = 0.20` | Default confidence threshold for detections |
| 9 | `DETECTION_CONF_POLE = 0.15` | Pole-specific confidence threshold |
| 10 | `DETECTION_IOU = 0.45` | NMS IoU threshold |
| 12–15 | `THRESHOLD_*` | Per-class thresholds (insulator, crossarm, pole, conductor) |
| 17–21 | `*_IMGSZ` | Image size per model stage (640 for pole, 1024 for others) |
| 27–40 | `OBB_CLASS_KEYWORDS` | Keyword routing dict mapping raw class names to canonical labels. `"vegetation"` key routes tree/plant/bush detections |
| 43–55 | `load_dotenv()` | Reads `.env` file into `os.environ` at import time |
| 60–68 | DB / app constants | `APP_SECRET_KEY`, `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASS`, `PG_DB` |
| 70–74 | `normalize_detection_label(label)` | Canonicalises detection labels: lowercase, underscores, no double underscores |

---

### `db.py` — 597 lines (relevant sections)

The tree feature does **not** have its own DB table. It uses:
- `uploaded_images` table (saves original uploaded images)
- `get_db_connection()` — returns a PostgreSQL connection via psycopg2
- `save_uploaded_image(name, mime, bytes, source_path)` — inserts into `uploaded_images`

| Lines | Symbol | Description |
|-------|--------|-------------|
| ~1–30 | Imports + connection setup | psycopg2 + `RealDictCursor` |
| `get_db_connection()` | Returns a `DBConn` adapter wrapping a psycopg2 connection |
| `save_uploaded_image()` | Stores image bytes and metadata; returns `id` |
| `init_postgres_db()` | Runs all `CREATE TABLE IF NOT EXISTS` DDL statements on startup |

---

### `image_processing.py` — 169 lines

The YOLO inference pipeline. Called by the vegetation context service; **not** called directly by `tree_routes.py` (the tree route sends images straight to PlantNet + Azure).

| Lines | Function | Description |
|-------|----------|-------------|
| 19–26 | Module init | Loads `InfrastructurePipeline` and `cable_model = YOLO(...)` at startup |
| 29–37 | `_run_raw_pipeline(temp_filename, log_mem)` | Runs `InfrastructurePipeline.predict()` on a saved temp file |
| 40–54 | `_build_cable_mask(temp_filename, height, width, log_mem)` | Runs cable YOLO segmentation; builds a binary mask of cable pixels |
| 57–66 | `_should_run_conductor_stage(final_detections, main_pole_boxes)` | Skips conductor extraction if no pole or structural hardware found |
| 69–99 | `_prepare_conductor_masks(mask_resized)` | Skeletonises + morphologically cleans the cable mask |
| 102–168 | `process_image_file(file_stream)` | **Main entry point** — saves temp file → runs pipeline → collects detections → conductor extraction → returns `{detections, master, survey_questionnaire, annotated_image, width, height}` |

---

### `detection_processing.py` — 299 lines

Post-processes raw YOLO outputs. Shared by the image pipeline.

| Lines | Function | Description |
|-------|----------|-------------|
| 8–125 | `merge_collinear_conductors(conductors, img_w, img_h)` | Merges collinear conductor segments: checks angle alignment (dot product ≥ 0.95), lateral offset (≤ 30 px), gap (≤ 35 % of image width); uses convex hull to combine polygons |
| 128–211 | `collect_model_detections(pipe_res, width, height)` | Collects insulators, crossarms, poles, street lights, and others from pipeline result; filters background detections by x-proximity to primary pole; builds hardware mask |
| 214–268 | `extract_conductor_detections(mask_closed, hardware_mask, main_pole_boxes, dist, skel, width, height)` | Finds conductor contours from cable mask; filters by aspect ratio, solidity, and overlap with pole; computes avg wire thickness from skeleton distance transform |
| 271–287 | `build_master_data(pipe_res)` | Builds top-level report dict from pipeline result (pole class, type, lean angle) |
| 290–298 | `build_survey_questionnaire(pipe_res, conductors)` | Builds survey summary (pole count, conductor count, vegetation flag) |

---

### `routes/auth_routes.py` — 78 lines

| Lines | Function | Description |
|-------|----------|-------------|
| 9–57 | `register_auth_routes(app)` | Registers `/login`, `/signup`, `/logout` routes with session management |
| 60–67 | `login_required(func)` | Decorator that redirects to `/login` if `session['user']` is absent — used on the tree detect endpoint |
| 70–77 | `admin_required(func)` | Decorator for admin-only routes (not used by tree detect) |

---

### `routes/tree_routes.py` — 536 lines  ← **Core file**

#### Utility Helpers
| Lines | Function | Description |
|-------|----------|-------------|
| 17–29 | `_safe_json_loads(text)` | Strips markdown code fences then `json.loads()`; used to parse Azure LLM responses |
| 32–34 | `_guess_mime(filename, fallback)` | Guesses MIME type from filename extension |
| 37–39 | `_image_bytes_to_data_url(image_bytes, mime_type)` | Base64-encodes image bytes to a `data:<mime>;base64,...` URL for Azure vision API |
| 42–44 | `_top_prediction(plantnet_result)` | Returns first entry of `top_predictions` list |
| 47–54 | `_best_common_name(prediction, plantnet_result)` | Picks best common name from PlantNet result, falls back to scientific name |
| 57–63 | `_species_confidence_label(score)` | Maps float score → `"high"` (≥0.75) / `"medium"` (≥0.45) / `"low"` |
| 66–89 | `_build_growth_estimate(growth_stage, probable_species)` | Returns a human-readable time-to-8m estimate based on species and growth stage |
| 92–111 | `_truncate_text(value, max_words)` | Trims to max word count with ellipsis |
| 103–111 | `_truncate_list(items, max_items, max_words)` | Trims a list to max items, each truncated |

#### PlantNet Species ID
| Lines | Function | Description |
|-------|----------|-------------|
| 114–160 | `identify_species_with_plantnet(image_bytes, filename, mime_type)` | Posts image to `https://my-api.plantnet.org/v2/identify/{project}`; returns `{best_match, best_score, top_predictions[:5], remaining_requests_today, predicted_organs}` |
| 115 | `PLANTNET_API_KEY` | Read from `os.getenv("PLANTNET_API_KEY")` |
| 116 | `PLANTNET_PROJECT` | Read from `os.getenv("PLANTNET_PROJECT", "all")` |
| 124 | URL | `https://my-api.plantnet.org/v2/identify/{project}?api-key=...&lang=en&nb-results=5` |

#### Azure OpenAI Config
| Lines | Function | Description |
|-------|----------|-------------|
| 163–186 | `_azure_runtime_configs()` | Builds list of Azure config dicts from env vars. Supports **primary** (`AZURE_OPENAI_*`) and **secondary** (`AZURE_OPENAI_mini_4.1_*`) deployments |
| 190–217 | `_run_azure_tree_completion(config, user_content, max_completion_tokens)` | Calls `AzureOpenAI.chat.completions.create()` with `response_format={"type":"json_object"}`; returns `(response, content, finish_reason)` |
| 220–251 | `_extract_token_usage(response, config, ...)` | Extracts prompt/completion/total tokens from response; computes per-image rates |

#### Azure LLM Tree Analysis
| Lines | Function | Description |
|-------|----------|-------------|
| 255–366 | `analyze_tree_with_azure_openai(full_tree_bytes, full_tree_mime_type, plantnet_result, leaf_bytes, leaf_mime_type)` | Sends 1–2 images to Azure GPT-4 Vision with a structured JSON prompt; retries up to 3 token budgets (1200 / 1600 / 2200) per deployment; raises `ValueError` with usage stats on total failure |
| 266–317 | Azure prompt | Requests: `headline`, `simple_verdict`, `top_match {scientific_name, common_name, confidence_percent}`, `technical_summary {probable_species, species_confidence, growth_stage, years_to_cross_8m, growth_direction, pole_impact_risk, pole_impact_note, overall_health, visible_signs, recommended_follow_up, limitations}` |

#### Result Building
| Lines | Function | Description |
|-------|----------|-------------|
| 370–402 | `build_fallback_tree_result(plantnet_result, warnings)` | Constructs a valid result dict from PlantNet data alone, used when Azure is unavailable |
| 405–453 | `normalize_tree_result(result, plantnet_result, warnings)` | Merges Azure result with fallback; enforces word limits via `_truncate_text` / `_truncate_list`; always includes `plantnet`, `warnings`, `generated_at` |

#### HTTP Endpoint
| Lines | Function | Description |
|-------|----------|-------------|
| 456–535 | `register_tree_routes(app)` → `POST /api/tree_detect` | **Main endpoint** — accepts multipart `image` + optional `leaf_image`; runs PlantNet → Azure LLM → `normalize_tree_result`; appends leaf-upload hint when no leaf provided; returns full JSON result |

---

## API Endpoint

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/tree_detect` | ✅ `login_required` | Identify tree species + analyse health/risk |

### Request — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `image` | ✅ | Full-tree JPEG / PNG / WebP |
| `leaf_image` | ➖ | Optional close-up leaf photo for higher species confidence |

### Response JSON

```jsonc
{
  "headline": "Tree scan completed: likely Blue Gum",
  "simple_verdict": "A single tree image was enough for a first-pass estimate.",
  "top_match": {
    "scientific_name": "Eucalyptus globulus",
    "common_name": "Blue Gum",
    "confidence_percent": 78
  },
  "technical_summary": {
    "probable_species": "Blue Gum",
    "species_confidence": "high",              // high | medium | low
    "growth_stage": "mature",                  // seedling | young | semi-mature | mature | old | unknown
    "years_to_cross_8m": "already near or above 8 meters",
    "growth_direction": "balanced",            // upward | sideward | balanced | unknown
    "pole_impact_risk": "medium",              // low | medium | high | unknown
    "pole_impact_note": "Crown spread may rub against nearby conductors.",
    "overall_health": "good — no visible disease or dieback",
    "visible_signs": ["Dense crown", "No dead branches", "Active foliage"],
    "recommended_follow_up": [
      "Schedule trimming within 6 months",
      "Upload one clear leaf photo for stronger species confirmation"
    ],
    "limitations": "Single image estimate only. Not survey-grade."
  },
  "plantnet": {
    "best_match": "Eucalyptus globulus Labill.",
    "best_score": 0.78,
    "top_predictions": [...],
    "remaining_requests_today": 490
  },
  "warnings": [],
  "generated_at": "2026-07-14T05:45:00Z",
  "token_usage": {
    "provider": "azure_openai",
    "status": "completed",         // completed | failed | not_used
    "deployment": "gpt-4o",
    "total_tokens": 1450,
    "prompt_tokens": 1200,
    "completion_tokens": 250,
    "image_count": 1
  }
}
```

---

## Data Flow

```
POST /api/tree_detect
  │
  ├─ tree_routes.py L459-535  (register_tree_routes → tree_detect)
  │
  ├─ 1. PlantNet species identification (L484-487)
  │      identify_species_with_plantnet(image_bytes, filename, mime_type)
  │      └─ POST https://my-api.plantnet.org/v2/identify/{project}
  │         returns top-5 species predictions + confidence scores
  │
  ├─ 2. Azure OpenAI vision analysis (L490-513)
  │      analyze_tree_with_azure_openai(full_tree_bytes, ..., plantnet_result, leaf_bytes)
  │      ├─ _azure_runtime_configs()          L163 — load primary + secondary configs
  │      ├─ _image_bytes_to_data_url()        L37  — encode images to base64 data URLs
  │      ├─ _run_azure_tree_completion()      L190 — call GPT-4 Vision (JSON mode)
  │      └─ _extract_token_usage()            L220 — parse token counts from response
  │
  ├─ 3. Result normalisation (L518)
  │      normalize_tree_result(llm_result, plantnet_result, warnings)
  │      ├─ build_fallback_tree_result()      L370 — PlantNet-only result if Azure fails
  │      ├─ _truncate_text() / _truncate_list()  L92/103 — enforce word limits
  │      └─ _build_growth_estimate()          L66  — years-to-8m estimate
  │
  └─ 4. JSON response returned to caller
```

---

## Required Environment Variables

Create a `.env` file at project root:

```ini
# Flask
APP_SECRET_KEY=your-secret-key

# PostgreSQL
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASS=yourpassword
PG_DB=gis_yolo

# PlantNet API
PLANTNET_API_KEY=your-plantnet-api-key
PLANTNET_PROJECT=all           # or a specific flora project slug

# Azure OpenAI — Primary deployment (GPT-4 Vision)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-primary-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2025-03-01-preview

# Azure OpenAI — Secondary / fallback deployment (optional)
AZURE_OPENAI_mini_4.1_ENDPOINT=https://your-secondary.openai.azure.com/
AZURE_OPENAI_mini_4.1_API_KEY=your-secondary-key
AZURE_OPENAI_mini_4.1_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_mini_4.1_mini_API_VERSION=2024-02-01
```

---

## Key Constants

| Constant | Value | File | Description |
|----------|-------|------|-------------|
| `ALLOWED_MIME_TYPES` | `{image/jpeg, image/png, image/jpg, image/webp}` | `tree_routes.py` L14 | Accepted image formats for PlantNet |
| Token budgets | `1200 → 1600 → 2200` | `tree_routes.py` L345 | Azure retry token limits per attempt |
| PlantNet `nb-results` | `5` | `tree_routes.py` L128 | Number of species predictions returned |
| `_guess_mime` fallback | `"image/jpeg"` | `tree_routes.py` L34 | Default MIME when filename gives no hint |
| `_truncate_text` headline | `max_words=14` | `tree_routes.py` L423 | Headline word limit |
| `_truncate_text` verdict | `max_words=22` | `tree_routes.py` L424 | Simple verdict word limit |
| `_truncate_list` visible_signs | `max_items=3` | `tree_routes.py` L442 | Max bullet points in visible signs |
| `_truncate_list` follow_up | `max_items=4` | `tree_routes.py` L443 | Max recommended follow-up items |
| Species confidence — high | score ≥ 0.75 | `tree_routes.py` L59 | Label mapping |
| Species confidence — medium | score ≥ 0.45 | `tree_routes.py` L61 | Label mapping |

---

## Minimal Standalone Implementation

To implement tree detection only, you need:

```
your_app/
├── .env                   # env vars above
├── app.py                 # trimmed to register only tree + auth routes
├── config.py              # full file (normalize_detection_label used by pipeline)
├── db.py                  # for get_db_connection + session/user tables
├── routes/
│   ├── __init__.py
│   ├── auth_routes.py     # login_required
│   └── tree_routes.py     # the full tree detection logic
└── requirements.txt       # key packages: flask, requests, openai, pillow
```

**Minimum pip installs:**
```
flask
requests
openai>=1.0
pillow
psycopg2-binary
werkzeug
```

The tree route does **not** depend on `image_processing.py`, `detection_processing.py`, `pipeline.py`, or any YOLO model — those are only used by the infrastructure detection pipeline.

---

## PlantNet API Notes

- Free tier: **500 requests/day** (tracked in response as `remaining_requests_today`)
- Endpoint: `https://my-api.plantnet.org/v2/identify/{project}`
- `organs=auto` lets PlantNet auto-detect whether the photo shows a leaf, flower, fruit, or bark
- `lang=en` returns English common names
- Only `image/jpeg`, `image/png`, `image/jpg`, `image/webp` are accepted (see `ALLOWED_MIME_TYPES`)

## Azure OpenAI Notes

- Requires a GPT-4 Vision capable deployment (e.g. `gpt-4o`, `gpt-4-vision-preview`)
- `response_format={"type":"json_object"}` enforces structured JSON output
- Primary + secondary configs are tried in order; within each, three increasing token budgets are tried before giving up
- `finish_reason='length'` triggers the next (larger) token budget; any other finish reason breaks the retry loop for that config
