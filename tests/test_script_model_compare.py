"""Modellsammenligningen på /ml-skript: fortløpende trening uten lekkasje, kjøp og hold, mål, tidsbudsjett og side."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import script_model_compare as C
import script_profit_ml as spm
import script_profit_predictor as predictor


def synthetic(n=1500, seed=4):
    """Daglige eksempler der gevinst bestemmes av egenskapen «a»; «b» og «c» er støy. Fasit kjent etter 7 døgn."""
    rng = np.random.default_rng(seed)
    decision = pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC")
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    win = (a + 0.3 * rng.normal(size=n) > 0.4).astype(int)
    ret = np.where(win == 1, 0.02, -0.02) + 0.001 * rng.normal(size=n)
    hold = 0.01 * rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "decision": decision, "known": decision + pd.Timedelta(days=7),
                         "ret": ret, "hold": hold, "excess": ret - hold, "win": win})


class DataTests(unittest.TestCase):
    def test_buy_and_hold_uses_the_same_opens_as_the_script_target(self):
        table = C.load_table("ut_bot", "60", "Neste uke")
        bars = pd.read_csv(predictor.ROOT / f"{C.FOLDER}/BTCUSDT_60_bars.csv", usecols=["open_time", "open"])
        opens = pd.Series(bars.open.to_numpy(), index=pd.to_datetime(bars.open_time, utc=True, format="ISO8601"))
        for i in (0, len(table) // 2, len(table) - 1):
            row = table.iloc[i]
            self.assertAlmostEqual(row.hold, opens[row.known] / opens[row.decision] - 1, places=12)
            self.assertAlmostEqual(row.excess, row.ret - row.hold, places=12)
            self.assertEqual(row.known - row.decision, pd.Timedelta(hours=168))
        self.assertTrue(((table.ret > 0).astype(int) == table.win).all())
        self.assertEqual(list(table.columns[:len(spm.ALL_COLUMNS)]), spm.ALL_COLUMNS)

    def test_changed_dataset_fails_closed(self):
        manifest, results = predictor.load()
        altered = {**manifest, "dataset_sha256": {k: "0" * 64 for k in manifest["dataset_sha256"]}}
        with patch.object(predictor, "load", return_value=(altered, results)), patch.dict(C._FILES, clear=True):
            with self.assertRaisesRegex(ValueError, "Datasettet er endret"):
                C.load_table("ut_bot", "60", "Neste uke")


class WalkForwardTests(unittest.TestCase):
    def test_training_only_uses_outcomes_known_before_each_refit_and_rolling_window_is_respected(self):
        table = C.load_table("chandelier", "240", "Neste måned")
        decision, known = C._ns(table.decision), C._ns(table.known)
        for window in (None, C.WINDOWS[C.ROLLING]):
            points = C.refit_points(table, "YS")
            runs = C.walk_forward(table, spm.MODEL_COLUMNS, [C.LR], points, window)
            blocks = runs[C.LR].blocks
            self.assertEqual(len(blocks), len(points))
            for start, block in zip(C._ns(points), blocks):
                self.assertTrue((decision[block.rows] >= start).all())
                self.assertTrue((known[block.train] <= start).all())
                self.assertGreaterEqual(len(block.train), C.MIN_TRAIN)
                if window is not None:
                    self.assertTrue((decision[block.train] >= start - window.value).all())
            before = table.decision < C.EVAL_START
            self.assertTrue(np.isnan(runs[C.LR].proba[before.to_numpy()]).all())
            self.assertFalse(np.isnan(runs[C.LR].proba[(~before).to_numpy()]).any())

    def test_models_learn_a_real_pattern_and_importance_finds_it(self):
        table = synthetic()
        points = pd.date_range(pd.Timestamp("2017-01-01", tz="UTC"), table.decision.max(), freq="QS")
        runs = C.walk_forward(table, ["a", "b", "c"], [C.LR, C.RF, C.HGB], points, None)
        mask = (table.decision >= "2017-01-01").to_numpy()
        for name, run in runs.items():
            rep = C.evaluate(table, run, mask)
            self.assertLess(rep["brier"], rep["baseline_brier"], name)
            self.assertGreater(rep["accuracy"], rep["baseline_accuracy"] + 0.1, name)
            self.assertLess(rep["mse_excess"], rep["baseline_mse_excess"], name)
            self.assertEqual(rep["fp"] + rep["fn"], round(rep["n"] * (1 - rep["accuracy"])))
        importance = C.permutation_importance(table, runs[C.RF], ["a", "b", "c"], mask)
        self.assertEqual(importance.index[0], "a")
        self.assertGreater(importance["a"], 0)
        latency, throughput = C.speed((runs[C.LR].blocks[-1].clf, runs[C.LR].blocks[-1].reg, 0.5),
                                      table[["a", "b", "c"]].to_numpy())
        self.assertGreater(latency, 0)
        self.assertGreater(throughput, 0)

    def test_time_budget_picks_slower_cadence_and_hard_limit_reuses_last_model(self):
        table = C.load_table("ut_bot", "1440", "Neste dag")
        self.assertEqual(C.choose_cadence(table, [1.0], None, 0.0), "Hver måned")
        self.assertEqual(C.choose_cadence(table, [600.0], None, 0.0), "Hvert halvår")
        self.assertEqual(C.choose_cadence(table, [10_000.0], None, 0.0), "Hvert år")
        points = C.refit_points(table, "YS")
        runs = C.walk_forward(table, spm.MODEL_COLUMNS, [C.LR], points, None, deadline=time.monotonic() - 1)
        blocks = runs[C.LR].blocks
        self.assertEqual(len(runs[C.LR].fit_ms), 1)
        self.assertEqual(sum(b.stale for b in blocks), len(blocks) - 1)
        self.assertFalse(np.isnan(runs[C.LR].proba[(table.decision >= C.EVAL_START).to_numpy()]).any())


class ViewTests(unittest.TestCase):
    def test_chosen_model_predicts_the_chosen_indicator_with_all_measures(self):
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Ingen HTTP")):
            out = C.view("kivanc_st", "60", "Neste dag", C.HGB, 5000, predictor.DEMO, "Hvert år")
        self.assertEqual(len(out), 6)
        text, predictions, quality, speed, drift, importance = out
        self.assertIn(f"{C.HGB} anslår", text)
        self.assertIn("SuperTrend [KivancOzbilgic] blir lønnsom", text)
        for word in ("meravkastning mot vanlig kjøp og hold", "Beregningstid", "lagrede, tunede modellen",
                     "finansiell rådgivning", "Indikatorens profitt mot vanlig kjøp og hold"):
            self.assertIn(word, text)
        for word in ("Hva målingene sier", "| Treffsikkerhet |", "★ = valgt modell"):   # fjernet etter ønske
            self.assertNotIn(word, text)
        self.assertEqual(len(predictions), len(C.available_models()))          # valgt modell + sammenligning
        self.assertEqual(predictions["Modell"].str.contains("★ valgt").sum(), 1)
        self.assertIn(f"{C.HGB} ★ valgt", list(predictions["Modell"]))
        self.assertIn("Meravkastning på 5 000 kr", predictions.columns)
        self.assertEqual(list(quality.columns), C.QUALITY_COLUMNS)
        self.assertFalse([c for c in quality.columns if "MSE" in c])            # fjernet etter ønske
        self.assertEqual(list(speed.columns), C.SPEED_COLUMNS)
        self.assertEqual(set(speed["Nye treninger"]), {"8"})
        self.assertIn(C.HGB, drift.layout.annotations[1].text)                   # kalibrering for valgt modell
        self.assertIn(C.HGB, importance.layout.title.text)
        again = C.view("kivanc_st", "60", "Neste dag", C.HGB, 5000, predictor.DEMO, "Hvert år")[0]
        self.assertIn("mellomlager", again)

    def test_single_model_without_comparison_and_other_feature_set(self):
        out = C.view("ut_bot", "240", "Neste uke", C.LR, 1000, predictor.DEMO, "Hvert år", C.ROLLING,
                     "Grunnleggende · 27 egenskaper", compare_all=False)
        self.assertEqual(len(out[1]), 1)
        self.assertEqual(len(out[2]), 4)                         # grunnlinje + valgt modell i to perioder
        self.assertEqual(len(out[3]), 1)
        self.assertIn("rullerende", out[0])

    def test_live_failure_keeps_history_but_gives_no_prediction_and_bad_input_is_refused(self):
        with patch.object(predictor, "live_input", side_effect=ValueError("Datakilden er utilgjengelig")), \
                patch.dict(C._LIVE, clear=True):
            text, predictions, quality, *rest = C.view("ut_bot", "60", "Neste dag", C.LR, 1000, predictor.LIVE,
                                                       "Hvert år", compare_all=False)
        self.assertIn("Ingen prediksjon for valgt input", text)
        self.assertTrue(predictions.empty)
        self.assertFalse(quality.empty)
        for args in ({"capital": -1}, {"capital": True}, {"model": "Ukjent modell"}, {"cadence": "hvert sekund"},
                     {"indicator": "vilkårlig pine"}):
            message = C.view(**({"model": C.LR, "cadence": "Hvert år", "compare_all": False} | args))[0]
            self.assertIn("Ingen ML-vurdering", message)


if __name__ == "__main__":
    unittest.main()
