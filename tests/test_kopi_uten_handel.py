"""Visningskopien kan ikke handle: ingen ordrekode, ingen nøkler, bare offentlige markedsdata og papirhandel."""
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import papirhandel as P
import weex_autotrade_dashboard as W

ROOT = Path(__file__).resolve().parents[1]
# Satt sammen av deler, så denne testfilen ikke selv inneholder ordene den leter etter.
FORBIDDEN = ["/capi/v3/" + "order", "placeTp" + "SlOrder", "close" + "Positions", "/capi/v3/" + "account",
             "ACCESS-" + "SIGN", "ACCESS-" + "KEY", "ACCESS-" + "PASSPHRASE", "WEEX_API_" + "KEY",
             "WEEX_API_" + "SECRET", "WEEX_API_" + "PASSPHRASE", "hmac." + "new", "weex_" + "trading",
             "weex_" + "connection", "carry_" + "trading", "LIVE_" + "CONFIRM", "api-" + "sim.weex"]
TEXT_SUFFIXES = {".py", ".md", ".json", ".pine", ".txt", ".cfg", ".toml", ".yml", ".yaml", ".csv"}
ALLOWED_ENV = {"GRADIO_SERVER_NAME", "GRADIO_SERVER_PORT", "WEEX_ENV"}


def bars(steps, end="2026-10-28 12:00", minutes=5):
    """Kunstige, avsluttede lys: ``steps`` er logavkastning per lys."""
    close = 30000 * np.exp(np.cumsum(steps))
    open_ = np.r_[close[0], close[:-1]]
    index = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=len(steps), freq=f"{minutes}min", tz="UTC",
                          name="open_time")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0005,
                         "low": np.minimum(open_, close) * 0.9995, "close": close, "volume": 1.0}, index=index)


def trend(n=600, seed=5):
    rng = np.random.default_rng(seed)
    return np.r_[np.full(n // 3, 0.002), np.full(n // 3, -0.002), np.full(n - 2 * (n // 3), 0.002)] + \
        rng.normal(0, 0.0005, n)


class NoNetwork:
    """Stopper ethvert nettverkskall i testene."""

    def __enter__(self):
        self.patches = [patch("requests.get", side_effect=AssertionError("Ingen nettverk i testen")),
                        patch("requests.sessions.Session.request", side_effect=AssertionError("Ingen nettverk"))]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in self.patches:
            item.stop()


class RepositoryTests(unittest.TestCase):
    def test_repository_contains_no_order_code_keys_or_account_data(self):
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or "__pycache__" in path.parts or not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            self.assertNotIn(path.suffix, {".sqlite3", ".db", ".env", ".pem", ".key"}, relative)
            self.assertNotEqual(path.name, ".env", relative)
            self.assertNotIn("runtime", relative.parts, relative)
            if path.suffix in TEXT_SUFFIXES and path != Path(__file__):
                text = path.read_text(encoding="utf-8", errors="ignore")
                for word in FORBIDDEN:
                    self.assertNotIn(word, text, f"{word} i {relative}")

    def test_code_reads_no_secret_environment_variables(self):
        found = set()
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                found |= set(re.findall(r'os\.environ[^"\n]*"([A-Z_]+)"', path.read_text(encoding="utf-8")))
        self.assertLessEqual(found, ALLOWED_ENV)


class PublicDataTests(unittest.TestCase):
    def test_public_client_only_sends_get_to_market_paths_without_credentials(self):
        calls = []

        class Reply:
            status_code = 200

            def json(self):
                return {"symbols": [{"symbol": "BTCUSDT", "pricePrecision": 1}]}

        with patch("requests.get", side_effect=lambda url, **kwargs: calls.append((url, kwargs)) or Reply()):
            W._CACHE.clear()
            self.assertEqual(W.price_step("BTCUSDT"), 0.1)
        url, kwargs = calls[0]
        self.assertTrue(url.startswith(W.MARKET_DOMAIN + "/capi/v3/market/"))
        self.assertFalse({k.lower() for k in kwargs["headers"]} &
                         {"access-key", "access-sign", "access-passphrase", "authorization"})
        self.assertTrue(all(path.startswith("/capi/v3/market/") for path in W.PUBLIC_PATHS))
        with self.assertRaises(W.MarketDataError):
            W._public_get("/capi/v3/" + "order")
        W._CACHE.clear()


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        P.reset()
        self.full = bars(trend())
        self.shown = 300
        self.market = patch.object(W, "market_bars",
                                   lambda environment=None, timeframe="5", symbol="BTCUSDT", limit=1000:
                                   self.full.iloc[:self.shown])
        self.step = patch.object(W, "price_step", lambda symbol: 0.1)
        self.history = patch.object(W, "with_history", lambda symbol, timeframe, frame, extra=1000: frame)
        for item in (self.market, self.step, self.history):
            item.start()

    def tearDown(self):
        for item in (self.market, self.step, self.history):
            item.stop()
        P.reset()

    def test_start_follows_the_script_reverses_and_closes_without_any_order(self):
        with NoNetwork():
            message = W.start_slot("#2", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter", ["SOL"],
                                   "Long og short (som UT Bot)", 10, 5, True)
            self.assertIn("Papirhandel startet", message)
            self.assertIn("allerede", W.start_slot("#1", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter",
                                                   ["SOL"], "Long og short (som UT Bot)", 10, 5, True))
            self.shown = 600
            status, states, fills, decisions = W.autotrade_view()
            self.assertIn("ingen ordre sendes", status)
            self.assertEqual(list(fills["Årsak"]), ["Inngang", "Snu", "Inngang (blir med ved start)"])
            self.assertEqual(list(fills["Posisjon"]), ["long", "short", "short"])
            self.assertTrue(all(fills["Papir-ID"].str.startswith("PAPIR-")))
            self.assertIn("LONG", states["Posisjon"].iloc[0])
            self.assertGreater(len(decisions), 0)
            qty = fills["Antall"].iloc[-1]
            self.assertAlmostEqual(qty * fills["Pris"].iloc[-1], 1000 * 0.10 * 5, delta=0.05)   # margin × gearing
            self.assertIn("lukket", W.close_slot("#2"))
            self.assertTrue(W.autotrade_view()[1].empty)

    def test_long_only_turns_short_signals_into_exits(self):
        with NoNetwork():
            W.start_slot("#3", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter", ["BTC"], "Kun long (SELG lukker)",
                         5, 2, True)
            self.shown = 600
            fills = W.autotrade_view()[2]
        self.assertNotIn("short", list(fills["Posisjon"]))

    def test_emergency_stop_at_90_percent_of_margin(self):
        steps = np.r_[np.full(200, 0.002), [-0.03], np.full(19, 0.0)]     # ett brått fall på 3 % (gearing 50 → stopp ved 1,8 %)
        self.full = bars(steps)
        self.shown = 200
        with NoNetwork():
            W.start_slot("#1", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter", ["BTC"],
                         "Long og short (som UT Bot)", 10, 50, True)
            self.shown = 220
            fills = W.autotrade_view()[2]
        stop = fills[fills["Årsak"].str.startswith("Nødstopp")]
        self.assertEqual(len(stop), 1)
        self.assertAlmostEqual(stop["Realisert (USDT)"].iloc[0], -0.9 * 100 - stop["Gebyr (USDT)"].iloc[0], places=1)

    def test_start_is_refused_without_acceptance_coins_or_valid_size(self):
        self.assertIn("Huk av", W.start_slot("#1", "Clean Wave Matrix PRO (Dark_Ace_Master)", "5 minutter", ["BTC"],
                                             "Long og short (som UT Bot)", 10, 5, False))
        self.assertIn("minst én", W.start_slot("#1", "Clean Wave Matrix PRO (Dark_Ace_Master)", "5 minutter", [],
                                               "Long og short (som UT Bot)", 10, 5, True))
        self.assertIn("Margin", W.start_slot("#1", "Clean Wave Matrix PRO (Dark_Ace_Master)", "5 minutter", ["BTC"],
                                             "Long og short (som UT Bot)", 50, 5, True))

    def test_every_script_draws_a_chart_from_public_bars_and_paper_fills_are_labelled(self):
        self.shown = 600
        with NoNetwork():
            for label in W.SLOT_STRATEGIES:
                text, figure, table = W.coin_chart_view("BTC", label, "5 minutter")
                self.assertIsNotNone(figure, label)
                self.assertNotIn("kunne ikke hentes", text, label)
            W.start_slot("#2", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter", ["BTC"],
                         "Long og short (som UT Bot)", 10, 5, True)
            text, figure, _ = W.coin_chart_view("BTC", "SuperTrend STRATEGY (KivancOzbilgic)", "5 minutter")
        names = [str(trace.name) for trace in figure.data]
        self.assertIn("Papirhandel (simulerte fyll)", names)
        self.assertFalse([name for name in names if "WEEX" in name and "fyll" in name])


class AppTests(unittest.TestCase):
    def test_main_page_has_the_fields_and_the_ml_page_answers_without_network(self):
        import gradio as gr
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import App
        self.assertEqual(App.demo.pages[0][1], "Automatisk handel · WEEX")
        with NoNetwork(), TestClient(gr.mount_gradio_app(FastAPI(), App.demo, path="/")) as client:
            config = client.get("/config").text
            for text in ("Start Skript #1", "Stopp #2", "Lukk #3", "Kryptovalutaer for Skript #2", "Gearing",
                         "Sanntidsgraf · Skript #3", "Utførte handler (papirfyll, simulert)", "papirhandel",
                         "TMG Dagsbekreftelse", "Kontroller WEEX-tilkobling"):
                self.assertIn(text, config)
            self.assertIn("Beregn med ML", client.get("/config", params={"page": "ml-skript"}).text)
            reply = client.post("/gradio_api/api/ml_script_assess", json={"data": [
                "ut_bot", "60", "Neste uke", "Logistisk regresjon / Ridge", 10000, "Lagret eksempel · 11. oktober 2026",
                "Hvert år", "Voksende · all kjent historikk", "Grunnleggende · 27 egenskaper", False]})
        self.assertEqual(reply.status_code, 200, reply.text[:200])
        self.assertIn("Logistisk regresjon / Ridge anslår", reply.json()["data"][0])


if __name__ == "__main__":
    unittest.main()
