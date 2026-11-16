"""
tests/test_survey_storage.py
Unit tests for Survey submission storage and auto-increment pole location number.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pole_autofill_service import store_submitted_survey, get_next_pole_location_number


class TestSurveyStorage(unittest.TestCase):

    def test_store_submitted_survey(self):
        payload = {
            "schema_id": "11kv_ht_pole",
            "title": "Survey - 11KV HT Pole",
            "pole_location_number": "P-999",
            "survey_data": {
                "pole_erection_status": "Pole with Conductor",
                "pole_location_number": "P-999",
                "pole_type": "Er. of 9.1M 280Kg WL w/o base plate"
            }
        }

        res = store_submitted_survey(payload)
        self.assertIn(res["status"], ["SUCCESS", "PARTIAL_SUCCESS"])
        self.assertIn("survey_id", res)
        self.assertEqual(res["pole_location_number"], "P-999")

    def test_auto_increment_pole_location_number(self):
        loc = get_next_pole_location_number()
        self.assertTrue(loc.startswith("P-"))
        self.assertEqual(len(loc), 5)


if __name__ == "__main__":
    unittest.main()
