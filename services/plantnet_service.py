from pathlib import Path

from routes.tree_routes import _guess_mime, identify_species_with_plantnet


def identify_species(image_path):
    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()
    mime_type = _guess_mime(image_path.name, 'image/jpeg')
    raw = identify_species_with_plantnet(image_bytes, image_path.name, mime_type)
    top_predictions = raw.get('top_predictions') or []
    top = top_predictions[0] if top_predictions else {}
    return {
        'best_match': raw.get('best_match') or top.get('scientific_name_without_author') or top.get('scientific_name') or '',
        'scientific_name': top.get('scientific_name_without_author') or top.get('scientific_name') or '',
        'common_name': ', '.join(top.get('common_names') or []) or top.get('scientific_name_without_author') or '',
        'confidence': float(top.get('score') or raw.get('best_score') or 0.0),
        'top_predictions': top_predictions[:5],
    }
