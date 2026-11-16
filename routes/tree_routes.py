import base64
import json
import mimetypes
import os
from datetime import datetime

import requests
from flask import jsonify, request
from openai import AzureOpenAI

from routes.auth_routes import login_required


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def _safe_json_loads(text):
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return json.loads(cleaned)


def _guess_mime(filename, fallback=None):
    guessed, _ = mimetypes.guess_type(filename or "")
    return guessed or fallback or "image/jpeg"


def _image_bytes_to_data_url(image_bytes, mime_type):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def _top_prediction(plantnet_result):
    predictions = (plantnet_result or {}).get("top_predictions", [])
    return predictions[0] if predictions else {}


def _best_common_name(prediction, plantnet_result):
    return (
        ", ".join(prediction.get("common_names") or [])
        or prediction.get("scientific_name_without_author")
        or prediction.get("scientific_name")
        or (plantnet_result or {}).get("best_match")
        or "Tree species not confidently identified"
    )


def _species_confidence_label(score):
    if isinstance(score, (int, float)):
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
    return "low"


def _build_growth_estimate(growth_stage, probable_species):
    stage = (growth_stage or "").lower()
    species = (probable_species or "").lower()

    if "mango" in species or "mangifera" in species:
        if "young" in stage:
            return "about 4-6 more years to cross 8 meters"
        if "semi" in stage:
            return "about 2-4 more years to cross 8 meters"
        if "mature" in stage:
            return "already near or above 8 meters, depending on pruning and variety"
        return "roughly 3-6 more years if growing under normal conditions"

    if "seedling" in stage:
        return "likely many years away from 8 meters"
    if "young" in stage:
        return "several more years to cross 8 meters"
    if "semi" in stage:
        return "possibly 2-5 more years to cross 8 meters"
    if "mature" in stage:
        return "may already be near 8 meters or could cross it soon"
    if "old" in stage:
        return "likely already above 8 meters or no longer in a strong height-growth phase"
    return "cannot estimate reliably from the current image set"


def _truncate_text(value, max_words=24):
    text = str(value or '').strip()
    if not text:
        return ''
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]).rstrip('.,;:') + '...'



def _truncate_list(items, max_items=4, max_words=12):
    if not isinstance(items, list):
        return []
    result = []
    for item in items[:max_items]:
        short = _truncate_text(item, max_words=max_words)
        if short:
            result.append(short)
    return result


def identify_species_with_plantnet(image_bytes, filename, mime_type):
    api_key = os.getenv("PLANTNET_API_KEY")
    project = os.getenv("PLANTNET_PROJECT", "all")

    if not api_key:
        raise ValueError("Missing PLANTNET_API_KEY")

    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type for Pl@ntNet: {mime_type}")

    url = f"https://my-api.plantnet.org/v2/identify/{project}"
    params = {
        "api-key": api_key,
        "lang": "en",
        "nb-results": 5,
        "include-related-images": "false",
    }
    files = [("images", (filename or "tree.jpg", image_bytes, mime_type))]
    data = {"organs": "auto"}

    response = requests.post(url, params=params, files=files, data=data, timeout=60)
    response.raise_for_status()

    raw = response.json()
    results = raw.get("results", [])
    top_predictions = []
    for item in results[:5]:
        species = item.get("species", {})
        top_predictions.append(
            {
                "score": item.get("score"),
                "scientific_name": species.get("scientificName"),
                "scientific_name_without_author": species.get("scientificNameWithoutAuthor"),
                "common_names": species.get("commonNames", []),
                "family": species.get("family", {}).get("scientificName"),
                "genus": species.get("genus", {}).get("scientificName"),
            }
        )

    return {
        "best_match": raw.get("bestMatch"),
        "best_score": top_predictions[0].get("score") if top_predictions else None,
        "remaining_requests_today": raw.get("remainingIdentificationRequests"),
        "predicted_organs": raw.get("predictedOrgans", []),
        "top_predictions": top_predictions,
        "raw_version": raw.get("version"),
    }


def _azure_runtime_configs():
    configs = []

    primary = {
        "name": "primary",
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }
    if primary["endpoint"] and primary["api_key"] and primary["deployment"]:
        configs.append(primary)

    secondary = {
        "name": "secondary",
        "endpoint": os.getenv("AZURE_OPENAI_mini_4.1_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_mini_4.1_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_mini_4.1_mini_API_VERSION", "2024-02-01"),
        "deployment": os.getenv("AZURE_OPENAI_mini_4.1_DEPLOYMENT"),
    }
    if secondary["endpoint"] and secondary["api_key"] and secondary["deployment"]:
        configs.append(secondary)

    return configs



def _run_azure_tree_completion(config, user_content, max_completion_tokens):
    client = AzureOpenAI(
        azure_endpoint=config["endpoint"],
        api_key=config["api_key"],
        api_version=config["api_version"],
    )

    response = client.chat.completions.create(
        model=config["deployment"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful plant survey assistant. "
                    "Return valid JSON only. "
                    "Be conservative and do not overclaim."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason or "unknown"
    return response, content, finish_reason


def _extract_token_usage(response, config, max_completion_tokens, finish_reason, image_count):
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    accepted_prediction_tokens = getattr(completion_details, "accepted_prediction_tokens", None) if completion_details else None
    rejected_prediction_tokens = getattr(completion_details, "rejected_prediction_tokens", None) if completion_details else None

    prompt_per_image = round(prompt_tokens / image_count, 2) if image_count else 0.0
    completion_per_image = round(completion_tokens / image_count, 2) if image_count else 0.0
    total_per_image = round(total_tokens / image_count, 2) if image_count else 0.0

    return {
        "provider": "azure_openai",
        "deployment": config.get("deployment"),
        "config_name": config.get("name"),
        "finish_reason": finish_reason,
        "max_completion_tokens": max_completion_tokens,
        "image_count": image_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_image": prompt_per_image,
        "completion_tokens_per_image": completion_per_image,
        "total_tokens_per_image": total_per_image,
        "reasoning_tokens": reasoning_tokens,
        "accepted_prediction_tokens": accepted_prediction_tokens,
        "rejected_prediction_tokens": rejected_prediction_tokens,
    }



def analyze_tree_with_azure_openai(full_tree_bytes, full_tree_mime_type, plantnet_result, leaf_bytes=None, leaf_mime_type=None):
    configs = _azure_runtime_configs()
    if not configs:
        raise ValueError("Azure OpenAI configuration is incomplete")

    leaf_guidance = (
        "A leaf image is also provided. Use it to improve species confidence if helpful."
        if leaf_bytes and leaf_mime_type
        else "No leaf image is provided. If leaf detail would improve confidence, ask for one in recommended_follow_up."
    )

    prompt = f"""
You are a tree analysis assistant for a field survey app.

Rules:
1. Prefer a single top species match only.
2. One full-tree image is usually enough. A second leaf image is optional and only improves confidence.
3. Predict growth stage from visible maturity: seedling | young | semi-mature | mature | old | unknown.
4. Give an estimated time to cross 8 meters only as a broad visual estimate, never exact.
5. Predict whether the tree is likely growing mainly upward, mainly sideward, or balanced from the visible crown and branching habit.
6. If the tree is beside a pole, estimate whether sideways growth is likely to threaten the pole or stay low-risk.
7. Do not include canopy density.
8. Do not include line clearance risk.
9. Do not create a separate 'For Everyone' block.
10. If a leaf image was not provided and it would help, mention that in follow-up.
11. Keep the response concise and practical.
12. Use short phrases, not long paragraphs.
13. Visible signs: maximum 3 bullets.
14. Recommended follow-up: maximum 4 bullets.
15. Overall health should be a short label with a brief note.
16. When describing growth direction, state whether the tree appears height-seeking, side-spreading, or balanced.
17. For pole impact, say clearly whether sideways growth is likely to press against, rub, or stay clear of a nearby pole.

Leaf-image guidance:
{leaf_guidance}

PlantNet candidate result:
{json.dumps(plantnet_result or {}, separators=(",", ":"))}

Return only valid JSON with this schema:
{{
  "headline": "",
  "simple_verdict": "",
  "top_match": {{
    "scientific_name": "",
    "common_name": "",
    "confidence_percent": 0
  }},
  "technical_summary": {{
    "probable_species": "",
    "species_confidence": "high | medium | low",
    "growth_stage": "seedling | young | semi-mature | mature | old | unknown",
    "years_to_cross_8m": "",
    "growth_direction": "upward | sideward | balanced | unknown",
    "pole_impact_risk": "low | medium | high | unknown",
    "pole_impact_note": "",
    "overall_health": "",
    "visible_signs": [""],
    "recommended_follow_up": [""],
    "limitations": ""
  }}
}}
"""

    user_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": _image_bytes_to_data_url(full_tree_bytes, full_tree_mime_type),
                "detail": "high",
            },
        },
    ]

    if leaf_bytes and leaf_mime_type:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_bytes_to_data_url(leaf_bytes, leaf_mime_type),
                    "detail": "high",
                },
            }
        )

    errors = []
    usage_attempts = []
    image_count = 2 if leaf_bytes and leaf_mime_type else 1
    for config in configs:
        for token_limit in (1200, 1600, 2200):
            try:
                response, content, finish_reason = _run_azure_tree_completion(config, user_content, token_limit)
                usage_stats = _extract_token_usage(response, config, token_limit, finish_reason, image_count)
                usage_attempts.append(usage_stats)
                parsed = _safe_json_loads(content)
                if isinstance(parsed, dict):
                    return parsed, usage_stats

                reasoning_tokens = usage_stats.get("reasoning_tokens")
                errors.append(
                    f'{config["name"]} deployment {config["deployment"]} returned no valid JSON '
                    f'(finish_reason={finish_reason}, max_completion_tokens={token_limit}, reasoning_tokens={reasoning_tokens})'
                )
                if finish_reason != 'length':
                    break
            except Exception as exc:
                errors.append(f'{config["name"]} deployment {config["deployment"]}: {exc}')
                break

    error_message = ' ; '.join(errors) if errors else 'Azure OpenAI did not return a usable result'
    raise ValueError(json.dumps({"message": error_message, "usage_attempts": usage_attempts}))



def build_fallback_tree_result(plantnet_result, warnings):
    prediction = _top_prediction(plantnet_result)
    common_name = _best_common_name(prediction, plantnet_result)
    best_score = prediction.get("score") or (plantnet_result or {}).get("best_score")
    score_text = f"{best_score * 100:.0f}%" if isinstance(best_score, (int, float)) else "limited"
    confidence_label = _species_confidence_label(best_score)
    default_stage = "mature" if isinstance(best_score, (int, float)) and best_score >= 0.75 else "unknown"

    return {
        "headline": f"Tree scan completed: likely {common_name}",
        "simple_verdict": "A single tree image was enough for a first-pass species and maturity estimate.",
        "top_match": {
            "scientific_name": prediction.get("scientific_name_without_author") or prediction.get("scientific_name") or "Unknown species",
            "common_name": common_name,
            "confidence_percent": int(round(best_score * 100)) if isinstance(best_score, (int, float)) else 0,
        },
        "technical_summary": {
            "probable_species": common_name,
            "species_confidence": confidence_label,
            "growth_stage": default_stage,
            "years_to_cross_8m": _build_growth_estimate(default_stage, common_name),
            "growth_direction": "balanced" if default_stage == "mature" else "unknown",
            "pole_impact_risk": "unknown",
            "pole_impact_note": "Cannot judge pole damage risk reliably without seeing the tree relative to the pole.",
            "overall_health": "visual screening only",
            "visible_signs": ["Species match confidence: " + score_text],
            "recommended_follow_up": [
                "If you want stronger species confirmation, upload one clear leaf photo",
                "Keep at least one full-tree image for the maturity estimate",
            ],
            "limitations": "Fallback mode was used because one or more external analysis services were unavailable.",
        },
    }


def normalize_tree_result(result, plantnet_result, warnings):
    fallback = build_fallback_tree_result(plantnet_result, warnings)
    if not isinstance(result, dict):
        result = {}

    top_match = result.get("top_match") if isinstance(result.get("top_match"), dict) else {}
    technical = result.get("technical_summary") if isinstance(result.get("technical_summary"), dict) else {}
    top_prediction = _top_prediction(plantnet_result)

    visible_signs = technical.get("visible_signs")
    if not isinstance(visible_signs, list) or not visible_signs:
        visible_signs = fallback["technical_summary"]["visible_signs"]

    recommended_follow_up = technical.get("recommended_follow_up")
    if not isinstance(recommended_follow_up, list) or not recommended_follow_up:
        recommended_follow_up = fallback["technical_summary"]["recommended_follow_up"]

    normalized = {
        "headline": _truncate_text(result.get("headline") or fallback["headline"], max_words=14),
        "simple_verdict": _truncate_text(result.get("simple_verdict") or fallback["simple_verdict"], max_words=22),
        "top_match": {
            "scientific_name": _truncate_text(top_match.get("scientific_name") or fallback["top_match"]["scientific_name"], max_words=6),
            "common_name": _truncate_text(top_match.get("common_name") or fallback["top_match"]["common_name"], max_words=6),
            "confidence_percent": top_match.get("confidence_percent") if isinstance(top_match.get("confidence_percent"), int) else fallback["top_match"]["confidence_percent"],
        },
        "technical_summary": {
            "probable_species": _truncate_text(technical.get("probable_species") or fallback["technical_summary"]["probable_species"], max_words=8),
            "species_confidence": (technical.get("species_confidence") or fallback["technical_summary"]["species_confidence"] or "low").lower(),
            "growth_stage": _truncate_text(technical.get("growth_stage") or fallback["technical_summary"]["growth_stage"], max_words=3),
            "years_to_cross_8m": _truncate_text(technical.get("years_to_cross_8m") or _build_growth_estimate(
                technical.get("growth_stage") or fallback["technical_summary"]["growth_stage"],
                technical.get("probable_species") or fallback["technical_summary"]["probable_species"],
            ), max_words=16),
            "growth_direction": (technical.get("growth_direction") or fallback["technical_summary"]["growth_direction"] or "unknown").lower(),
            "pole_impact_risk": (technical.get("pole_impact_risk") or fallback["technical_summary"]["pole_impact_risk"] or "unknown").lower(),
            "pole_impact_note": _truncate_text(technical.get("pole_impact_note") or fallback["technical_summary"]["pole_impact_note"], max_words=22),
            "overall_health": _truncate_text(technical.get("overall_health") or fallback["technical_summary"]["overall_health"], max_words=18),
            "visible_signs": _truncate_list(visible_signs, max_items=3, max_words=8),
            "recommended_follow_up": _truncate_list(recommended_follow_up, max_items=4, max_words=12),
            "limitations": _truncate_text(technical.get("limitations") or fallback["technical_summary"]["limitations"], max_words=18),
        },
        "plantnet": {
            **(plantnet_result or {}),
            "top_predictions": [top_prediction] if top_prediction else [],
        },
        "warnings": warnings,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return normalized


def register_tree_routes(app):
    @app.route('/api/tree_detect', methods=['POST'])
    @login_required
    def tree_detect():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        uploaded = request.files['image']
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({'error': 'Uploaded image is empty'}), 400

        filename = uploaded.filename or 'tree.jpg'
        mime_type = uploaded.mimetype or _guess_mime(filename, 'image/jpeg')

        leaf_uploaded = request.files.get('leaf_image')
        leaf_bytes = None
        leaf_mime_type = None
        if leaf_uploaded:
            leaf_bytes = leaf_uploaded.read()
            if leaf_bytes:
                leaf_mime_type = leaf_uploaded.mimetype or _guess_mime(leaf_uploaded.filename or 'leaf.jpg', 'image/jpeg')

        warnings = []
        plantnet_result = {}
        llm_result = None
        plantnet_warning = None

        try:
            plantnet_result = identify_species_with_plantnet(image_bytes, filename, mime_type)
        except Exception as exc:
            plantnet_warning = f'PlantNet unavailable: {exc}'

        token_usage = None
        try:
            llm_result, token_usage = analyze_tree_with_azure_openai(
                image_bytes,
                mime_type,
                plantnet_result,
                leaf_bytes=leaf_bytes,
                leaf_mime_type=leaf_mime_type,
            )
        except Exception as exc:
            exc_text = str(exc)
            try:
                exc_payload = json.loads(exc_text)
            except Exception:
                exc_payload = None
            if isinstance(exc_payload, dict):
                token_usage = {
                    "provider": "azure_openai",
                    "status": "failed",
                    "image_count": 2 if leaf_bytes and leaf_mime_type else 1,
                    "attempts": exc_payload.get("usage_attempts") or [],
                }
                warnings.append(f"Azure analysis unavailable: {exc_payload.get('message') or exc_text}")
            else:
                warnings.append(f'Azure analysis unavailable: {exc_text}')

        if llm_result is None and plantnet_warning:
            warnings.append(plantnet_warning)

        final_result = normalize_tree_result(llm_result, plantnet_result, warnings)
        if token_usage:
            if "status" not in token_usage:
                token_usage["status"] = "completed"
            final_result["token_usage"] = token_usage
        else:
            final_result["token_usage"] = {
                "provider": "azure_openai",
                "status": "not_used",
                "image_count": 2 if leaf_bytes and leaf_mime_type else 1,
            }
        if not leaf_bytes:
            follow_up = final_result["technical_summary"].get("recommended_follow_up") or []
            leaf_hint = "Upload one clear leaf photo if you want stronger species confirmation"
            if leaf_hint not in follow_up:
                follow_up.append(leaf_hint)
            final_result["technical_summary"]["recommended_follow_up"] = follow_up
        return jsonify(final_result)
