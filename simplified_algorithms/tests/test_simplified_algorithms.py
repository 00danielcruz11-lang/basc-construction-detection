import unittest

from simplified_algorithms.construction_start import (
    CONFIRMED,
    PROVISIONAL,
    IndexStats,
    causal_onset,
    compare_series,
    compare_to_baseline,
    previous_period,
)
from simplified_algorithms.thermal_anomaly import ThermalObservation, detect_thermal_steps


class ConstructionStartTests(unittest.TestCase):
    def test_previous_period(self):
        self.assertEqual(previous_period("2026-03-H2"), "2026-03-H1")
        self.assertEqual(previous_period("2026-03-H1"), "2026-02-H2")

    def test_index_delta_rule(self):
        base = IndexStats("2026-01-H1", ndbi=0.10, swir=0.20, ndvi=0.40)
        cur = IndexStats("2026-01-H2", ndbi=0.121, swir=0.231, ndvi=0.31)
        call = compare_to_baseline(cur, base)
        self.assertTrue(call.changed)
        self.assertEqual(call.status, "ok")

    def test_causal_confirmation_dates_first_changed_look(self):
        series = [
            IndexStats("2026-01-H1", ndbi=0.08, swir=0.20),
            IndexStats("2026-01-H2", ndbi=0.08, swir=0.20),
            IndexStats("2026-02-H1", ndbi=0.08, swir=0.20),
            IndexStats("2026-02-H2", ndbi=0.13, swir=0.25),
            IndexStats("2026-03-H1", ndbi=0.17, swir=0.29),
        ]
        calls = causal_onset(compare_series(series))
        self.assertEqual(calls[-2].state, PROVISIONAL)
        self.assertEqual(calls[-1].state, CONFIRMED)
        self.assertEqual(calls[-1].confirmed_period, "2026-02-H2")


class ThermalAnomalyTests(unittest.TestCase):
    def test_detects_positive_step(self):
        rows = []
        for day in range(1, 25):
            date = f"2026-01-{day:02d}"
            for n in range(1, 15):
                building = f"B{n:02d}"
                bt = 303.0 + (0.1 if n % 2 else 0.0)
                if building == "B13" and day >= 14:
                    bt += 2.5
                rows.append(ThermalObservation(building, date, bt, clear_pixels=16, total_pixels=16))

        results = {
            result.building: result
            for result in detect_thermal_steps(rows, min_peers=10, min_segment=6, permutations=300)
        }
        self.assertEqual(results["B13"].verdict, "thermal_rise")
        self.assertEqual(results["B13"].onset_date, "2026-01-14")


if __name__ == "__main__":
    unittest.main()
