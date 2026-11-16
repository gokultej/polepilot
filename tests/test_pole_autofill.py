"""
tests/test_pole_autofill.py
Unit tests for the 39-question Auto-Fill service and 33KV conditional visibility.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pole_autofill_service import populate_survey_answers, get_next_pole_location_number


class TestPoleAutofillService(unittest.TestCase):

    def test_next_pole_location_number_format(self):
        loc_num = get_next_pole_location_number()
        self.assertTrue(loc_num.startswith("P-"))
        self.assertEqual(len(loc_num), 5)  # Format: P-001

    def test_autofill_11kv_ht_pole_shed_count_3(self):
        payload = {
            "detections": [
                {"label": "ht_pole"},
                {"label": "conductor"},
                {"label": "ht_pin"},
                {"label": "ht_pin"},
                {"label": "shed"},
                {"label": "shed"},
                {"label": "shed"},  # 3 sheds
                {"label": "stay_set"},
                {"label": "cleat"},
                {"label": "ab_switch"},
                {"label": "t_raising_crossarm"},
                {"label": "channel"}
            ],
            "ocr_results": {"text": "P-11/402"}
        }

        result = populate_survey_answers(payload)
        answers = result["answers"]

        # Verification of auto-filled fields
        self.assertEqual(answers["pole_erection_status"], "Pole with Conductor")
        self.assertEqual(answers["pole_to_pole_distance"], "")
        self.assertEqual(result['field_provenance']['pole_to_pole_distance'], 'unavailable_requires_measurement')
        self.assertEqual(answers["is_pin_point"], "Yes")
        self.assertEqual(answers["pole_type"], "Er. of 9.1M 280Kg WL w/o base plate")
        self.assertEqual(answers["er_of_stay_set_complete_exc_concrete"], "1")
        self.assertEqual(answers["no_of_channels_erected"], "2")  # t_raising + channel = 2
        self.assertEqual(answers["er_of_11kv_polymer_insulators_with_pins"], 2)
        self.assertEqual(answers["er_of_11kv_top_cleat_with_clamp_wt_4_69k"], 1)
        self.assertEqual(answers["erection_of_200amps_ab_switch_horizontal_type"], 1)
        self.assertEqual(answers["numbering_of_pole_with_ac_black_paint"], "Yes")

        # Verify 33KV section is hidden when shed_count <= 3
        self.assertFalse(result["is_33kv_section_visible"])

    def test_autofill_33kv_section_visible_when_shed_count_gt_3(self):
        payload = {
            "detections": [
                {"label": "ht_pole"},
                {"label": "ht_pin"},
                {"label": "shed"},
                {"label": "shed"},
                {"label": "shed"},
                {"label": "shed"},  # 4 sheds -> 33KV
                {"label": "v_crossarm"}
            ]
        }

        result = populate_survey_answers(payload)
        answers = result["answers"]

        # Verify 33KV section is visible when shed_count > 3
        self.assertTrue(result["is_33kv_section_visible"])
        self.assertEqual(result["shed_count"], 4)
        self.assertEqual(answers["er_of_33kv_v_channel_tapping_channel_x_arm"], 1)


if __name__ == "__main__":
    unittest.main()
