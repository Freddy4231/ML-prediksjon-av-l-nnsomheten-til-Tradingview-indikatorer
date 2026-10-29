"""Maskinlæring på Pine-skriptets handler: bare kjent historikk, fortløpende trening uten lekkasje, mål og side."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pine_engine import ml as ML
from pine_engine import runner
from pine_engine.broker import Trade

T0 = 1_735_689_600_000          # 2025-01-01 00:00 UTC
HOUR = 3_600_000


def bars(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    t = T0 + HOUR * np.arange(n, dtype=np.int64)
    return runner.Bars("BTCUSDT", 60, t, t + HOUR, open_, np.maximum(open_, close) * 1.002,
                       np.minimum(open_, close) * 0.998, close, rng.uniform(1, 10, n), tickerid="BINANCE:BTCUSDT")


def trades_with_rule(b, rule, every=12, hold=6, start=120):
    """Handler der gevinst/tap bestemmes av ``rule(features på beslutningslyset)``."""
    features = ML.market_features(b)
    out = []
    for k, entry in enumerate(range(start, len(b.C) - hold - 1, every)):
        decision = entry - 1
        win = rule(features.iloc[decision])
        qty, price = 1.0, float(b.O[entry])
        pnl = qty * price * (0.01 if win else -0.01)
        out.append(Trade(1 if k % 2 == 0 else -1, entry, price, entry + hold, float(b.C[entry + hold]), qty, pnl,
                         0.0, 0.0, "L", "signal"))
    return out


class FeatureTests(unittest.TestCase):
    def test_features_use_only_the_bar_and_earlier_bars(self):
        b = bars(600)
        full = ML.market_features(b)
        cut = 400
        short = runner.Bars(b.symbol, 60, b.T[:cut + 1], b.TC[:cut + 1], b.O[:cut + 1], b.H[:cut + 1], b.L[:cut + 1],
                            b.C[:cut + 1], b.V[:cut + 1], tickerid=b.tickerid)
        pd.testing.assert_series_equal(full.iloc[cut], ML.market_features(short).iloc[cut], check_names=False)

    def test_trade_rows_use_the_signal_bar_and_only_closed_trades(self):
        b = bars(800)
        trades = [Trade(1, 200, 100.0, 260, 101.0, 1.0, 1.0, 0, 0, "L", "x"),    # avsluttes etter neste beslutning
                  Trade(-1, 230, 100.0, 240, 99.0, 1.0, 1.0, 0, 0, "S", "x"),
                  Trade(1, 300, 100.0, 310, 98.0, 1.0, -2.0, 0, 0, "L", "x")]
        table = ML.trade_table(b, trades)
        self.assertEqual(list(table.decision), [199, 229, 299])
        self.assertEqual(table.prev_return.iloc[1], 0.0)             # første handel var ikke avsluttet ved 229
        self.assertAlmostEqual(table.prev_return.iloc[2], 0.01)       # siste avsluttede før 299: handelen fra 230
        self.assertEqual(list(table.win), [1, 1, 0])


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.b = bars()
        self.trades = trades_with_rule(self.b, lambda f: f["ret_12"] > 0)
        self.table = ML.trade_table(self.b, self.trades)

    def test_training_never_uses_trades_closed_after_the_decision(self):
        for mode in ("expanding", "rolling"):
            wf = ML.walk_forward(self.table, ML.LR, mode=mode, window=50, refit=7, min_train=30)
            self.assertTrue(wf.blocks)
            for blk in wf.blocks:
                self.assertTrue((self.table.exit_bar.to_numpy()[blk.train_rows]
                                 <= self.table.decision.to_numpy()[blk.rows[0]]).all())
                if mode == "rolling":
                    self.assertLessEqual(len(blk.train_rows), 50)
            self.assertTrue(np.isnan(wf.proba[:30]).all())               # ingen prediksjon før 30 kjente utfall

    def test_time_budget_trains_less_often_but_never_on_unknown_trades(self):
        free = ML.walk_forward(self.table, ML.RF, refit=3, min_train=30)
        tight = ML.walk_forward(self.table, ML.RF, refit=3, min_train=30, budget=0.3)
        self.assertFalse(free.adjusted)
        self.assertTrue(tight.adjusted)
        self.assertEqual(tight.requested, 3)
        self.assertGreater(tight.step, 3)
        self.assertLess(len(tight.blocks), len(free.blocks))
        self.assertTrue(np.isnan(tight.proba[:30]).all())
        self.assertFalse(np.isnan(tight.proba[30:]).any())          # alle senere handler får fortsatt en prediksjon
        for blk in tight.blocks:
            self.assertTrue((self.table.exit_bar.to_numpy()[blk.train_rows]
                             <= self.table.decision.to_numpy()[blk.rows[0]]).all())
        rep = ML.evaluate(self.table, tight)
        self.assertTrue(rep["adjusted"])
        self.assertEqual(rep["requested_step"], 3)

    def test_models_learn_a_real_pattern_and_importance_finds_it(self):
        result = ML.run(self.b, self.trades, models=(ML.LR, ML.RF, ML.HGB), refit=10, min_train=30,
                        deadline=time.monotonic() + 300)
        for name in (ML.LR, ML.RF, ML.HGB):
            rep = result["reports"][name]
            self.assertLess(rep["mse"], rep["baseline_mse"], name)
            self.assertGreater(rep["accuracy"], 0.75, name)
            self.assertEqual(rep["tp"] + rep["fp"] + rep["fn"] + rep["tn"], rep["predicted"])
            self.assertGreater(rep["latency_ms"], 0)
            self.assertGreater(rep["throughput"], 0)
        self.assertEqual(result["importance"][ML.LR].index[0], "ret_12")
        self.assertIn("ret_12", list(result["native"][ML.RF].index[:3]))
        self.assertTrue(any("slår grunnraten" in tip or "Beste modell" in tip for tip in result["tips"]))

    def test_feature_selection_keeps_the_useful_feature(self):
        table = self.table.iloc[:150].reset_index(drop=True)
        keep = ML._select_features(ML.LR, table[ML.FEATURES], table.win.to_numpy(), ML.FEATURES)
        self.assertIn("ret_12", keep)
        self.assertLess(len(keep), len(ML.FEATURES))

    def test_unpredictable_trades_do_not_beat_the_base_rate_in_the_advice(self):
        rng = np.random.default_rng(1)
        noise = trades_with_rule(self.b, lambda f: rng.random() < 0.4)
        result = ML.run(self.b, noise, models=(ML.LR,), refit=10, min_train=30, deadline=time.monotonic() + 120)
        rep = result["reports"][ML.LR]
        if rep["mse"] >= rep["baseline_mse"]:
            self.assertTrue(any("Ingen modell" in tip for tip in result["tips"]))

    def test_metrics_and_filter_result(self):
        table = pd.DataFrame({"win": [1, 0, 1, 0], "ret": [0.1, -0.05, 0.02, -0.01], "decision": [1, 2, 3, 4],
                              "exit_bar": [2, 3, 4, 5], "entry_time": [0, 1, 2, 3]})
        wf = ML.WalkForward("x", np.array([0.9, 0.8, 0.2, np.nan]), np.array([0.5, 0.5, 0.5, np.nan]))
        wf.blocks.append(ML.Block(None, np.arange(3), [], np.arange(1), 0.5, 1.0, 1.0, 0.1))
        rep = ML.evaluate(table, wf)
        self.assertEqual((rep["predicted"], rep["tp"], rep["fp"], rep["fn"], rep["tn"]), (3, 1, 1, 1, 0))
        self.assertAlmostEqual(rep["mse"], ((0.9 - 1) ** 2 + 0.8 ** 2 + 0.8 ** 2) / 3)
        self.assertAlmostEqual(rep["filtered"], 1.1 * 0.95 - 1)
        self.assertAlmostEqual(rep["fp_cost"], -0.05)
        self.assertAlmostEqual(rep["fn_missed"], 0.02)

    def test_too_few_trades_is_explained(self):
        with self.assertRaises(ValueError):
            ML.run(self.b, self.trades[:10])


class PageTests(unittest.TestCase):
    def test_page_runs_backtest_and_returns_all_outputs(self):
        import pine_dashboard as page
        b = bars()
        trades = trades_with_rule(b, lambda f: f["ret_12"] > 0)
        fake = {"raw": {"bars": b, "first": 0, "trades": trades}}
        with patch.object(page.B, "run_timeframe", return_value=fake) as backtest:
            summary, table, importance, drift, errors = page.run_ml(
                "//@version=5\nstrategy('x')\n", "binance:BTCUSDT", "Siste år", "", "", "Long og short", 0.1, True,
                True, page.B.AUTO, page.B.AUTO, None, "1 time", [ML.LR, ML.RF], list(page.ML_MODES)[1], 100, 10, 30,
                0.5, True, progress=lambda *a, **k: None)
        self.assertEqual(backtest.call_args.args[2], 60)
        self.assertTrue(backtest.call_args.kwargs["keep_raw"])
        self.assertIn("slår grunnraten", summary)
        self.assertTrue(table["visible"] and importance["visible"] and drift["visible"])
        self.assertEqual(list(table["value"].columns), page.ML_COLUMNS)
        self.assertEqual(len(table["value"]), 4)                     # to modeller, med og uten feature-utvalg
        message = page.run_ml("", "binance:BTCUSDT", "Siste år", "", "", "Long og short", 0.1, True, True, None, None,
                              None, "1 time", [ML.LR], list(page.ML_MODES)[0], 100, 5, 30, 0.5, False)[0]
        self.assertIn("Lim inn", message)


if __name__ == "__main__":
    unittest.main()
