"""
tests/test_pole_classifier.py
Unit tests for the 8 pole survey classification rules.
"""

import sys
import os
import unittest

# Add workspace path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pole_classifier_service import classify_pole_survey


class TestPoleClassifierService(unittest.TestCase):

    def test_pole_not_detected(self):
        payload = {"detections": []}
        res = classify_pole_survey(payload)
        self.assertFalse(res["pole_detected"])
        self.assertEqual(res["status"], "POLE_NOT_DETECTED")

    def test_rule_1_11kv_ht_pole(self):
        payload = {
            "detections": [
                {"label": "ht_pole", "confidence": 0.95},
                {"label": "ht_pin"},
                {"label": "ht_pin"},
                {"label": "shed"},
                {"label": "shed"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "11kv_ht_pole")

    def test_rule_2_ht_lt_pole(self):
        payload = {
            "detections": [
                {"label": "pole", "confidence": 0.95},
                {"label": "ht_pin"},
                {"label": "lt_pin"},
                {"label": "lt_pin"},
                {"label": "tapping"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "ht_lt_pole")
        self.assertIn("Tapping Channel", res["reason"])

    def test_rule_3_dtr_and_pole(self):
        payload = {
            "detections": [
                {"label": "pole", "confidence": 0.95},
                {"label": "transformer"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "dtr_and_pole")

    def test_rule_4_6_3kv_to_11kv_pole(self):
        payload = {
            "detections": [
                {"label": "pole", "confidence": 0.95},
                {"label": "ht_pin"},
                {"label": "shed"},
                {"label": "shed"},
                {"label": "shed"},
                {"label": "shed"}  # 4 sheds
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "6_3kv_to_11kv_pole")

    def test_rule_5_lt_1_phase_3_wire_pole(self):
        payload = {
            "detections": [
                {"label": "lt_pole", "confidence": 0.95},
                {"label": "lt_pin"},
                {"label": "lt_pin"},
                {"label": "3_wire_crossarm"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "lt_1_phase_3_wire_pole")

    def test_rule_6_lt_3_phase_5_wire_pole(self):
        payload = {
            "detections": [
                {"label": "lt_pole", "confidence": 0.95},
                {"label": "lt_pin"},
                {"label": "lt_pin"},
                {"label": "lt_pin"},
                {"label": "lt_pin"},
                {"label": "5_wire_crossarm"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "lt_3_phase_5_wire_pole")

    def test_rule_7_lt_2x16_25_ab_cable_pole(self):
        payload = {
            "detections": [
                {"label": "pole", "confidence": 0.95},
                {"label": "3_wire_crossarm"},
                {"label": "3_wire_crossarm"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "lt_2x16_25_ab_cable_pole")

    def test_rule_8_lt_3x16_25_ab_cable_pole(self):
        payload = {
            "detections": [
                {"label": "pole", "confidence": 0.95},
                {"label": "5_wire_crossarm"},
                {"label": "5_wire_crossarm"}
            ]
        }
        res = classify_pole_survey(payload)
        self.assertTrue(res["pole_detected"])
        self.assertEqual(res["schema_id"], "lt_3x16_25_ab_cable_pole")


if __name__ == "__main__":
    unittest.main()
