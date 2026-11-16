"""Lightweight output guard shared by replay, live jobs and saved results."""
from config import is_pole_label, meets_pole_confidence


def accepted_poles(result):
    return [d for d in result.get('detections', []) if is_pole_label(d.get('label')) and meets_pole_confidence(d.get('confidence'))]


def satisfies_pole_policy(result):
    for detection in result.get('detections', []):
        if is_pole_label(detection.get('label')) and not meets_pole_confidence(detection.get('confidence')):
            return False
    if result.get('classification', {}).get('pole_detected') and not accepted_poles(result):
        return False
    if is_pole_label(result.get('master', {}).get('pole_type')) and not accepted_poles(result):
        return False
    return True
