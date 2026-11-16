"""
services/pole_classifier_service.py
Standalone pole classification and survey schema redirection engine.
Classifies detection outputs into 1 of 8 pole survey schemas or POLE_NOT_DETECTED.
"""

from typing import Dict, Any, List
from services.pole_autofill_service import populate_survey_answers
from config import filter_pole_detections, is_pole_label


def classify_pole_survey(detection_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies a pole based on YOLO detection results, bounding boxes, pin types,
    shed counts, crossarm types, and DTR presence.

    Rules:
    1. 11kv_ht_pole: Only HT pins detected and shed count <= 3.
    2. ht_lt_pole: Combinations of HT pins and LT pins on same pole.
       - 4 or 5 LT pins + tapping => 5 wire cross arm
       - 1, 2, or 3 LT pins + tapping => 3 wire cross arm
    3. dtr_and_pole: DTR / Transformer detected with respect to pole.
    4. 6_3kv_to_11kv_pole: HT pins detected, shed count >= 4 (4 or 5), no LT pins, no DTR.
    5. lt_1_phase_3_wire_pole: Only LT pins (1 to 3), no HT pins. 3 wire cross arm present.
    6. lt_3_phase_5_wire_pole: Only LT pins (4 or 5 with tapping), no HT pins. 5 wire cross arm present.
    7. lt_2x16_25_ab_cable_pole: 3 wire cross arm detected 2 times.
    8. lt_3x16_25_ab_cable_pole: 5 wire cross arm detected 2 times.
    9. POLE_NOT_DETECTED: No pole detected.
    """
    detections: List[Dict[str, Any]] = filter_pole_detections(detection_payload.get("detections", []))
    detection_payload = {**detection_payload, 'detections': detections}
    
    # Extract feature counts
    poles = [d for d in detections if is_pole_label(d.get('label'))]
    ht_pins = [d for d in detections if "ht_pin" in d.get("label", "").lower()]
    lt_pins = [d for d in detections if "lt_pin" in d.get("label", "").lower() or "shackle" in d.get("label", "").lower()]
    sheds = [d for d in detections if any(k in d.get("label", "").lower() for k in ["shed", "disc", "ht_disc", "disc_insulator"])]
    dtrs = [d for d in detections if any(k in d.get("label", "").lower() for k in ["transformer", "dtr"])]
    crossarms = [d for d in detections if "crossarm" in d.get("label", "").lower() or "cross" in d.get("label", "").lower() or "channel" in d.get("label", "").lower()]
    tappings = [d for d in detections if "tapping" in d.get("label", "").lower()]

    pole_count = len(poles)
    if pole_count == 0:
        return {
            "status": "POLE_NOT_DETECTED",
            "pole_detected": False,
            "schema_id": None,
            "title": "Pole Not Detected",
            "message": "No pole meets the minimum 80% model confidence. Please provide a clearer image or use a manually verified survey."
        }

    ht_pin_count = len(ht_pins)
    lt_pin_count = len(lt_pins)
    dtr_count = len(dtrs)
    has_tapping = len(tappings) > 0

    # Calculate shed_count: If both ht_pin and ht_disc present, evaluate both highest-conf insulators; otherwise evaluate the single active HT insulator.
    ht_pins_list = [d for d in detections if "ht_pin" in d.get("label", "").lower()]
    ht_discs_list = [d for d in detections if any(k in d.get("label", "").lower() for k in ["ht_disc", "disc_insulator"])]

    target_insulators = []
    if ht_pins_list and ht_discs_list:
        target_insulators = [
            max(ht_pins_list, key=lambda d: d.get("confidence", 0.0)),
            max(ht_discs_list, key=lambda d: d.get("confidence", 0.0))
        ]
    elif ht_pins_list:
        target_insulators = [max(ht_pins_list, key=lambda d: d.get("confidence", 0.0))]
    elif ht_discs_list:
        target_insulators = [max(ht_discs_list, key=lambda d: d.get("confidence", 0.0))]

    shed_count = 0
    if target_insulators:
        for ins in target_insulators:
            best_bbox = ins.get("bbox", [])
            if best_bbox and len(best_bbox) == 4:
                x1, y1, x2, y2 = best_bbox
                margin_x = (x2 - x1) * 0.20
                margin_y = (y2 - y1) * 0.20
                bx1, by1, bx2, by2 = x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y

                sub_sheds = [
                    d for d in detections
                    if any(k in d.get("label", "").lower() for k in ["shed", "disc"])
                    and len(d.get("bbox", [])) == 4
                    and bx1 <= ((d["bbox"][0] + d["bbox"][2]) / 2.0) <= bx2
                    and by1 <= ((d["bbox"][1] + d["bbox"][3]) / 2.0) <= by2
                ]

                if len(sub_sheds) > 0:
                    shed_count += len(sub_sheds)
                else:
                    shed_count += int(ins.get("shed_count", len(sheds) if len(sheds) > 0 else 1))
            else:
                shed_count += int(ins.get("shed_count", len(sheds) if len(sheds) > 0 else 1))
    else:
        shed_count = len(sheds)

    # Crossarm classifications
    count_3_wire_crossarms = 0
    count_5_wire_crossarms = 0

    for ca in crossarms:
        label = ca.get("label", "").lower()
        if "3_wire" in label or "3_phase" in label:
            count_3_wire_crossarms += 1
        elif "5_wire" in label or "5_phase" in label:
            count_5_wire_crossarms += 1

    # Infer crossarm type if not explicitly labeled
    if count_3_wire_crossarms == 0 and count_5_wire_crossarms == 0:
        if 1 <= lt_pin_count <= 3:
            count_3_wire_crossarms = 1
        elif 4 <= lt_pin_count <= 5:
            count_5_wire_crossarms = 1

    schema_id = None
    title = ""
    reason = ""

    # Rule 8: LT 3x16+25 AB Cable pole (5 wire cross arm detected 2 times)
    if count_5_wire_crossarms >= 2:
        schema_id = "lt_3x16_25_ab_cable_pole"
        title = "Survey - LT 3x16+25 AB Cable Pole"
        reason = "Detected 5-wire crossarm 2 or more times"

    # Rule 7: LT 2x16+25 AB Cable pole (3 wire cross arm detected 2 times)
    elif count_3_wire_crossarms >= 2:
        schema_id = "lt_2x16_25_ab_cable_pole"
        title = "Survey - LT 2x16+25 AB Cable Pole"
        reason = "Detected 3-wire crossarm 2 or more times"

    # Rule 3: DTR and Pole
    elif dtr_count > 0:
        schema_id = "dtr_and_pole"
        title = "Survey - DTR and Pole Structure"
        reason = f"Detected {dtr_count} DTR/Transformer component(s) near pole"

    # Rule 2: HT/LT Pole (Combination of HT pins and LT pins)
    elif ht_pin_count > 0 and lt_pin_count > 0:
        schema_id = "ht_lt_pole"
        title = "Survey - HT/LT Combined Pole"
        if has_tapping and ht_pin_count > 0:
            crossarm_desc = "Tapping Channel"
        elif lt_pin_count >= 4 and has_tapping:
            crossarm_desc = "5 wire cross arm"
        else:
            crossarm_desc = "3 wire cross arm"
        reason = f"Detected HT pins ({ht_pin_count}) and LT pins ({lt_pin_count}) with {crossarm_desc}"

    # Rule 1: 11KV HT Pole (Only HT pins, shed count <= 3)
    elif ht_pin_count > 0 and lt_pin_count == 0 and shed_count <= 3:
        schema_id = "11kv_ht_pole"
        title = "Survey - 11KV HT Pole"
        reason = f"Only HT pins detected ({ht_pin_count}) with shed count {shed_count} (<= 3)"

    # Rule 4: 6.3KV to 11KV Pole (Only HT pins, shed count >= 4)
    elif ht_pin_count > 0 and lt_pin_count == 0 and shed_count >= 4:
        schema_id = "6_3kv_to_11kv_pole"
        title = "Survey - 6.3KV to 11KV Pole"
        reason = f"Only HT pins detected ({ht_pin_count}) with shed count {shed_count} (>= 4)"

    # Rule 6: LT 3 Phase 5 Wire Pole (All LT pins, 4 or 5 LT pins + tapping)
    elif lt_pin_count >= 4 and ht_pin_count == 0:
        schema_id = "lt_3_phase_5_wire_pole"
        title = "Survey - LT 3 Phase 5 Wire Pole"
        reason = f"Only LT pins detected ({lt_pin_count}) with 5-wire crossarm configuration"

    # Rule 5: LT 1 Phase 3 Wire Pole (All LT pins, 1 to 3 LT pins)
    elif lt_pin_count > 0 and ht_pin_count == 0:
        schema_id = "lt_1_phase_3_wire_pole"
        title = "Survey - LT 1 Phase 3 Wire Pole"
        reason = f"Only LT pins detected ({lt_pin_count}) with 3-wire crossarm configuration"

    # Fallback default if pole is detected but no specific pins detected
    else:
        schema_id = "11kv_ht_pole"
        title = "Survey - 11KV HT Pole"
        reason = "Defaulted to 11KV HT Pole survey schema based on general pole detection"

    autofill_data = populate_survey_answers(detection_payload)

    return {
        "status": "SUCCESS",
        "pole_detected": True,
        "schema_id": schema_id,
        "title": title,
        "reason": reason,
        "autofill_data": autofill_data,
        "feature_summary": {
            "pole_count": pole_count,
            "ht_pin_count": ht_pin_count,
            "lt_pin_count": lt_pin_count,
            "shed_count": shed_count,
            "dtr_count": dtr_count,
            "has_tapping": has_tapping,
            "count_3_wire_crossarms": count_3_wire_crossarms,
            "count_5_wire_crossarms": count_5_wire_crossarms
        }
    }
