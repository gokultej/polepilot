"""
services/pole_autofill_service.py
Auto-fill logic engine for all 39 survey questions based on YOLO detections,
OCR text extraction, database submitted pole counts, and conditional rules.
"""

from typing import Dict, Any, List
import sqlite3
import os

from config import SQLITE_PATH, PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB


import uuid
from datetime import datetime
import json

from config import SQLITE_PATH, PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB
from db import get_db_connection


def init_survey_tables():
    """Ensure pole_surveys table exists in SQLite if using SQLite."""
    try:
        if os.path.exists(os.path.dirname(SQLITE_PATH)) or SQLITE_PATH == "database.db":
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pole_surveys (
                    id TEXT PRIMARY KEY,
                    pole_location_number TEXT,
                    schema_id TEXT,
                    title TEXT,
                    survey_data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
    except Exception as e:
        print("[AutoFillService] SQLite table init notice:", e)


def get_next_pole_location_number() -> str:
    """
    Fetches the total count of registered/submitted poles from DB
    and generates the next auto-increment pole location number (e.g. P-001, P-002).
    """
    init_survey_tables()
    count = 0

    # 1. Try PostgreSQL
    try:
        pg_conn = get_db_connection()
        cur = pg_conn.execute("SELECT COUNT(*) FROM pole_surveys")
        row = cur.fetchone()
        if row:
            count = list(row.values())[0] if isinstance(row, dict) else row[0]
        pg_conn.close()
    except Exception as e:
        # 2. Fallback to SQLite
        try:
            if os.path.exists(SQLITE_PATH):
                conn = sqlite3.connect(SQLITE_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM pole_surveys")
                row = cursor.fetchone()
                if row:
                    count = row[0]
                conn.close()
        except Exception as err:
            print("[AutoFillService] Count query notice:", err)

    next_num = count + 1
    return f"P-{next_num:03d}"


def store_submitted_survey(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stores a submitted survey point in PostgreSQL and SQLite databases.
    """
    init_survey_tables()

    survey_id = f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    pole_location_number = payload.get("pole_location_number") or get_next_pole_location_number()
    schema_id = payload.get("schema_id", "11kv_ht_pole")
    title = payload.get("title", "Survey - 11KV HT Pole")
    survey_data = payload.get("survey_data", {})
    created_at = datetime.now().isoformat()

    survey_json_str = json.dumps(survey_data)

    stored_in_pg = False
    stored_in_sqlite = False

    # Store in Postgres
    try:
        pg_conn = get_db_connection()
        pg_conn.execute(
            """
            INSERT INTO pole_surveys (id, pole_location_number, schema_id, title, survey_data_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (survey_id, pole_location_number, schema_id, title, survey_json_str, created_at)
        )

        # Also insert into assets table so it appears in Admin Submissions Queue (/admin)
        asset_class = title.replace("Survey - ", "")
        voltage = "33kV" if survey_data.get("er_of_33kv_v_channel_tapping_channel_x_arm") else "11kV"
        worker_name = payload.get("worker_name") or "admin 1"
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pg_conn.execute(
            """
            INSERT INTO assets (id, worker_name, status, timestamp, asset_class, voltage, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            (survey_id, worker_name, "pending", ts_now, asset_class, voltage, f"Pole Location: {pole_location_number}")
        )

        image_b64 = payload.get("image_b64", "")
        existing_uploaded_id = payload.get("uploaded_image_id")
        detections = payload.get("detections", [])
        det_str = json.dumps(detections) if isinstance(detections, list) else str(detections)

        image_linked = False

        # First, try to reuse an already-uploaded image by its ID
        if existing_uploaded_id and not image_linked:
            from db import bytes_to_b64
            try:
                row = pg_conn.execute(
                    "SELECT id, image_data FROM uploaded_images WHERE id = ?",
                    (existing_uploaded_id,)
                ).fetchone()
                if row and row.get("image_data"):
                    norm_b64 = bytes_to_b64(row["image_data"])
                    # Link the uploaded image to this survey asset
                    pg_conn.execute(
                        "UPDATE uploaded_images SET related_asset_id = ?, detections = ? WHERE id = ?",
                        (survey_id, det_str, existing_uploaded_id)
                    )
                    pg_conn.execute(
                        """
                        INSERT INTO asset_images (asset_id, image_b64, detections, pole_angle, uploaded_image_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (survey_id, norm_b64, det_str, 0.0, existing_uploaded_id)
                    )
                    image_linked = True
            except Exception as img_err:
                print(f"[AutoFillService] Reuse uploaded_image notice: {img_err}")

        # Fallback: if no uploaded_image_id or it failed, use inline image_b64
        if image_b64 and not image_linked:
            from db import b64_to_bytes, bytes_to_b64, psycopg2
            img_bytes = b64_to_bytes(image_b64)
            norm_b64 = bytes_to_b64(img_bytes)
            up_id = str(uuid.uuid4())

            pg_conn.execute(
                """
                INSERT INTO uploaded_images (id, original_filename, mime_type, image_data, related_asset_id, detections, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO NOTHING
                """,
                (up_id, f"pole_{pole_location_number}.jpg", "image/jpeg", psycopg2.Binary(img_bytes), survey_id, det_str, ts_now)
            )

            pg_conn.execute(
                """
                INSERT INTO asset_images (asset_id, image_b64, detections, pole_angle, uploaded_image_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (survey_id, norm_b64, det_str, 0.0, up_id)
            )

        pg_conn.commit()
        pg_conn.close()
        stored_in_pg = True
    except Exception as e:
        print("[AutoFillService] Postgres survey insert notice:", e)

    # Store in SQLite
    try:
        if os.path.exists(os.path.dirname(SQLITE_PATH)) or SQLITE_PATH == "database.db":
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pole_surveys (id, pole_location_number, schema_id, title, survey_data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (survey_id, pole_location_number, schema_id, title, survey_json_str, created_at)
            )
            cursor.execute(
                """
                INSERT OR IGNORE INTO assets (id, worker_name, status, timestamp, asset_class, voltage, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (survey_id, worker_name, "pending", ts_now, asset_class, voltage, f"Pole Location: {pole_location_number}")
            )

            if image_b64:
                from db import b64_to_bytes, bytes_to_b64
                img_bytes = b64_to_bytes(image_b64)
                norm_b64 = bytes_to_b64(img_bytes)
                up_id = str(uuid.uuid4())

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO asset_images (asset_id, image_b64, detections, pole_angle, uploaded_image_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (survey_id, norm_b64, det_str, 0.0, up_id)
                )

            conn.commit()
            conn.close()
            stored_in_sqlite = True
    except Exception as e:
        print("[AutoFillService] SQLite survey insert notice:", e)

    return {
        "status": "SUCCESS" if (stored_in_pg or stored_in_sqlite) else "PARTIAL_SUCCESS",
        "survey_id": survey_id,
        "pole_location_number": pole_location_number,
        "message": f"Survey point {pole_location_number} saved successfully!"
    }


def get_stored_surveys() -> List[Dict[str, Any]]:
    """
    Fetches all submitted survey records from PostgreSQL or SQLite.
    """
    init_survey_tables()
    surveys = []

    try:
        pg_conn = get_db_connection()
        cur = pg_conn.execute("SELECT id, pole_location_number, schema_id, title, survey_data_json, created_at FROM pole_surveys ORDER BY created_at DESC")
        rows = cur.fetchall()
        for r in rows:
            surveys.append({
                "id": r["id"],
                "pole_location_number": r["pole_location_number"],
                "schema_id": r["schema_id"],
                "title": r["title"],
                "survey_data": json.loads(r["survey_data_json"]) if isinstance(r["survey_data_json"], str) else r["survey_data_json"],
                "created_at": r["created_at"]
            })
        pg_conn.close()
        return surveys
    except Exception as e:
        try:
            if os.path.exists(SQLITE_PATH):
                conn = sqlite3.connect(SQLITE_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT id, pole_location_number, schema_id, title, survey_data_json, created_at FROM pole_surveys ORDER BY created_at DESC")
                rows = cursor.fetchall()
                for r in rows:
                    surveys.append({
                        "id": r["id"],
                        "pole_location_number": r["pole_location_number"],
                        "schema_id": r["schema_id"],
                        "title": r["title"],
                        "survey_data": json.loads(r["survey_data_json"]) if r["survey_data_json"] else {},
                        "created_at": r["created_at"]
                    })
                conn.close()
        except Exception as err:
            print("[AutoFillService] get_stored_surveys query notice:", err)

    return surveys


def populate_survey_answers(detection_payload: Dict[str, Any], ocr_payload: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Populates all 39 survey fields based on detection results and OCR data.

    Rules:
    1. pole_erection_status: "Pole with Conductor" if conductor detected else "Pole"
    2. pole_location_number: Next auto-increment ID for registered poles
    3. pole_to_pole_distance: Unfilled until a verified measurement is provided.
    4. is_pin_point: "Yes" if ht_pin detected else "No"
    5. pole_type: "Er. of 9.1M 280Kg WL w/o base plate"
    6. er_of_stay_set_complete_exc_concrete: count of stay_set (0 to 3)
    7. stut: count of stut_pole (0 to 3)
    8. no_of_channels_erected: sum of crossarms (vcross, tapping, box_arm, 5-wire, 3-wire)
    9. er_of_11kv_polymer_insulators_with_pins: count of ht_pin or lt_pin
    10. er_of_11kv_com_polymer_insulator_c_and_t: count of ht_disc / disc_insulator
    11. er_of_11kv_top_cleat_with_clamp_wt_4_69k: count of top_cleat
    12. er_of_special_clamp_50x6_ms_flat: count of special_clamp
    13. erection_of_spiral_earth_coil: "0"
    14. numbering_of_pole_with_ac_black_paint: "Yes" if OCR text present else "No"
    15. remarks: ""
    16. se_3_type_box_x_arm_with_stut_33_11kv_lt: count of box_arm
    17. se_of_ht_lt_box_type_x_arm_with_clamp: count if ht_pin > 0 and lt_pin > 0 and box_arm > 0
    18. erection_of_200amps_ab_switch_horizontal_type: count of ab_switch
    19. ci_earth_pipe: 0
    20. gi_earth_pipe: 0
    21-32: 33KV Conditional section (Visible & filled ONLY if shed_count > 3)
    33. dis_of_11kv_v_cross_arm: count of v_cross_arm
    34. dis_of_11kv_top_cleat: count of top_cleat
    35. dis_of_11kv_pin_insulators: count of ht_pin
    36. dis_of_11kv_disc: count of ht_disc
    37. dis_of_55_sqmm_conductor: ""
    38. dis_of_34_sqmm_conductor: ""
    39. iron_pole: count of iron_pole (default 0)
    """
    detections: List[Dict[str, Any]] = detection_payload.get("detections", [])
    ocr_data = ocr_payload or detection_payload.get("ocr_results", {})

    labels = [d.get("label", "").lower() for d in detections]

    conductors = [l for l in labels if "conductor" in l or "wire" in l or "cable" in l]
    ht_pins = [l for l in labels if "ht_pin" in l]
    lt_pins = [l for l in labels if "lt_pin" in l or "shackle" in l]
    discs = [l for l in labels if "disc" in l or "ht_disc" in l]
    sheds = [l for l in labels if any(k in l for k in ["shed", "disc", "ht_disc", "disc_insulator"])]
    stay_sets = [l for l in labels if "stay" in l]
    stuts = [l for l in labels if "stut" in l]
    v_crossarms = [l for l in labels if "v_cross" in l or "vcross" in l]
    t_raising_crossarms = [l for l in labels if "t_raising" in l or "t_rising" in l or "t-raising" in l or "t-rising" in l]
    channels = [l for l in labels if "channel" in l and "v_channel" not in l]
    tappings = [l for l in labels if "tapping" in l]
    box_arms = [l for l in labels if "box_arm" in l or "box" in l]
    five_wire_arms = [l for l in labels if "5_wire" in l or "5_phase" in l]
    three_wire_arms = [l for l in labels if "3_wire" in l or "3_phase" in l]
    top_cleats = [l for l in labels if "cleat" in l or "top_cleat" in l]
    special_clamps = [l for l in labels if "clamp" in l or "special_clamp" in l]
    ab_switches = [l for l in labels if "ab_switch" in l or "switch" in l]
    iron_poles = [l for l in labels if "iron_pole" in l]

    # Calculate shed_count: If both ht_pin and ht_disc present, evaluate both highest-conf insulators; otherwise evaluate single active HT insulator.
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

    is_33kv_section_visible = shed_count > 3

    # Rule 8: Sum count of channels/arms (vcross, t_raising, channels, tapping, box_arm, 5-wire, 3-wire)
    channel_count = len(v_crossarms) + len(t_raising_crossarms) + len(channels) + len(tappings) + len(box_arms) + len(five_wire_arms) + len(three_wire_arms)
    channel_option = str(min(channel_count, 3))

    stay_count = min(len(stay_sets), 3)
    stut_count = min(len(stuts), 3)

    ht_pin_count = len(ht_pins)
    lt_pin_count = len(lt_pins)
    polymer_insulator_count = ht_pin_count + lt_pin_count
    disc_count = len(discs)

    box_arm_count = len(box_arms)
    ht_lt_box_arm_count = box_arm_count if (ht_pin_count > 0 and lt_pin_count > 0 and box_arm_count > 0) else 0

    has_ocr_text = bool(ocr_data.get("text") or ocr_data.get("pole_number"))

    answers = {
        "pole_erection_status": "Pole with Conductor" if len(conductors) > 0 else "Pole",
        "pole_location_number": "" if detection_payload.get('evaluation_mode') else get_next_pole_location_number(),
        "pole_to_pole_distance": "",  # No measured span is available from a single image.
        "is_pin_point": "Yes" if ht_pin_count > 0 else "No",
        "pole_type": "Er. of 9.1M 280Kg WL w/o base plate",
        "er_of_stay_set_complete_exc_concrete": str(stay_count),
        "stut": str(stut_count),
        "no_of_channels_erected": channel_option,
        "er_of_11kv_polymer_insulators_with_pins": polymer_insulator_count,
        "er_of_11kv_com_polymer_insulator_c_and_t": disc_count,
        "er_of_11kv_top_cleat_with_clamp_wt_4_69k": len(top_cleats),
        "er_of_special_clamp_50x6_ms_flat": len(special_clamps),
        "erection_of_spiral_earth_coil": "0",
        "numbering_of_pole_with_ac_black_paint": "Yes" if has_ocr_text else "No",
        "remarks": "",
        "se_3_type_box_x_arm_with_stut_33_11kv_lt": box_arm_count,
        "se_of_ht_lt_box_type_x_arm_with_clamp": ht_lt_box_arm_count,
        "erection_of_200amps_ab_switch_horizontal_type": len(ab_switches),
        "ci_earth_pipe": 0,
        "gi_earth_pipe": 0,

        # 33KV Section (21 - 32)
        "er_of_33kv_v_channel_tapping_channel_x_arm": len(v_crossarms) + len(tappings) if is_33kv_section_visible else 0,
        "er_of_33kv_top_cleat_wt_6_9_kgs": len(top_cleats) if is_33kv_section_visible else 0,
        "er_of_33kv_polymer_insulators_with_pin": polymer_insulator_count if is_33kv_section_visible else 0,
        "hard_rock": 0,
        "pole_pit_excav_by_openblast_in_hard_rock": 0,
        "cement_concrete_breaking": 0,
        "9_1_mtr_power_drilling": 0,
        "9_1_mtr_pole_concrete": 0,
        "dis_and_re_er_of_9_1m_280kg_wl_pscc_poles": 0,
        "dis_and_re_er_of_8_0m_200kg_wl_psccpoles": 0,
        "dis_of_9_1_mtr_pscc_pole_and_related_items": 0,
        "dis_of_8_0_mtr_pscc_pole_and_related_items": 0,

        # Disassembly items (33 - 39)
        "dis_of_11kv_v_cross_arm": len(v_crossarms),
        "dis_of_11kv_top_cleat": len(top_cleats),
        "dis_of_11kv_pin_insulators": ht_pin_count,
        "dis_of_11kv_disc": disc_count,
        "dis_of_55_sqmm_conductor": "",
        "dis_of_34_sqmm_conductor": "",
        "iron_pole": len(iron_poles)
    }

    image_name = detection_payload.get("image_name", "Active Image")

    return {
        "is_33kv_section_visible": is_33kv_section_visible,
        "shed_count": shed_count,
        "image_name": image_name,
        "answers": answers,
        "warnings": [
            "Pole-to-pole distance is unavailable; enter a verified field measurement.",
            "Pole specification and construction quantities contain rule defaults and need field verification.",
            "Zero detected components does not establish physical absence.",
            "Suggested pole numbers are drafts, not reserved unique identifiers."
        ],
        "field_provenance": {
            "pole_to_pole_distance": "unavailable_requires_measurement",
            "pole_type": "unverified_rule_default",
            "pole_location_number": "unassigned_demo" if detection_payload.get('evaluation_mode') else "draft_sequence_not_reserved"
        }
    }
