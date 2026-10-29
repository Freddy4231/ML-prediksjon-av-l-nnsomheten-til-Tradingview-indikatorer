"""Etterligning av TradingViews strategimotor for lønnsomhetstesten.

- Markedsordre lagt ved et lys' close fylles ved neste åpning (``process_orders_on_close=true`` gir
  fyll ved samme close).
- ``pyramiding = 1``: ny inngang i samme retning ignoreres, motsatt inngang snur posisjonen.
- Hver posisjon bruker hele egenkapitalen (1×, ingen giring). Skriptets ``qty`` brukes ikke, så alle
  skript sammenlignes likt.
- ``strategy.exit`` (stop/limit/profit/loss/trailing, ``qty_percent``) sjekkes inne i lyset med
  TradingViews antakelse: ligger åpningen nærmest high, går prisen først til high, så low.
  Kurshopp forbi nivået fylles til åpningskurs.
- Kostnad per side av omsatt beløp, faktisk funding hver 8. time (long betaler positiv rate).
- Egenkapital ≤ 0 stopper all handel (konkurs).
"""
from dataclasses import dataclass

from pine_engine.runtime import NAN, isna, t_


@dataclass
class Trade:
    side: int
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    qty: float
    pnl: float
    fees: float
    funding: float
    entry_id: str
    reason: str


def _direction(value):
    if value is True or value == "long":
        return 1
    if value is False or value == "short":
        return -1
    return 1 if t_(value) else -1


class NullBroker:
    """Brukes i indikatorer og i hjelpekjøringer for request.security: alle ordre ignoreres."""
    pos = 0.0
    initial = 1.0

    def avg_price(self):
        return NAN

    def opentrades(self):
        return 0.0

    def closedtrades(self):
        return 0.0

    def equity_now(self, i):
        return 1.0

    def netprofit(self):
        return 0.0

    def openprofit(self, i):
        return 0.0

    def wintrades(self):
        return 0.0

    def losstrades(self):
        return 0.0

    def grossprofit(self):
        return 0.0

    def grossloss(self):
        return 0.0

    def entry_name(self):
        return ""

    def entry(self, *args):
        return None

    order = exit = close = close_all = cancel = cancel_all = allow_entry_in = entry


class Broker(NullBroker):
    def __init__(self, bars, first, cost=0.001, funding=None, long_only=False, on_close=False, mintick=0.01):
        self.O, self.H, self.L, self.C = bars.O, bars.H, bars.L, bars.C
        self.n, self.first = len(bars.C), first
        self.cost, self.funding = cost, funding
        self.long_only, self.fill_on_close, self.mintick = long_only, on_close, mintick
        self.cash = 1.0
        self.pos = 0.0
        self.avg = NAN
        self.entry_qty = 0.0
        self.entry_bar = -1
        self.entry_id = ""
        self.trade_fees = 0.0
        self.trade_funding = 0.0
        self.market = []           # ventende markedsordre fra forrige close
        self.entry_orders = {}     # id -> (retning, limit, stop)
        self.exits = {}            # id -> dict
        self.trades = []
        self.equity = [NAN] * self.n
        self.fees = 0.0
        self.funding_paid = 0.0
        self.bankrupt = False
        self.allowed = 0                   # strategy.risk.allow_entry_in: 0 = begge, 1 = long, -1 = short
        self.exposed_bars = 0

    # ------------------------------------------------ tall for skriptet

    def avg_price(self):
        return self.avg if self.pos else NAN

    def opentrades(self):
        return 1.0 if self.pos else 0.0

    def closedtrades(self):
        return float(len(self.trades))

    def equity_now(self, i):
        return self.cash + (self.pos * (self.C[i] - self.avg) if self.pos else 0.0)

    def netprofit(self):
        return self.cash - 1.0

    def openprofit(self, i):
        return self.pos * (self.C[i] - self.avg) if self.pos else 0.0

    def wintrades(self):
        return float(sum(1 for t in self.trades if t.pnl > 0))

    def losstrades(self):
        return float(sum(1 for t in self.trades if t.pnl < 0))

    def grossprofit(self):
        return sum(t.pnl for t in self.trades if t.pnl > 0)

    def grossloss(self):
        return -sum(t.pnl for t in self.trades if t.pnl < 0)

    def entry_name(self):
        return self.entry_id if self.pos else ""

    # ------------------------------------------------ ordre fra skriptet

    def _active(self, i, when):
        return t_(when) and i >= self.first and not self.bankrupt

    def entry(self, i, id_, direction, limit=NAN, stop=NAN, when=True):
        if not self._active(i, when):
            return None
        side = _direction(direction)
        if not isna(limit) or not isna(stop):
            self.entry_orders[str(id_)] = (side, limit, stop)
            return None
        self.market.append(("entry", str(id_), side, NAN))
        if self.fill_on_close:
            self._fill_market(i, self.C[i])
        return None

    order = entry

    def exit(self, i, id_, from_entry="", qty_percent=NAN, profit=NAN, limit=NAN, loss=NAN, stop=NAN,
             trail_price=NAN, trail_points=NAN, trail_offset=NAN, when=True):
        if not self._active(i, when):
            return None
        source = "" if isna(from_entry) else str(from_entry)
        key = (str(id_), source)          # samme exit-navn kan gjelde flere innganger (f.eks. «long» og «short»)
        previous = self.exits.get(key)
        state = {"from": source,
                 "qp": 100.0 if isna(qty_percent) else float(qty_percent),
                 "profit": profit, "limit": limit, "loss": loss, "stop": stop, "trail_price": trail_price,
                 "trail_points": trail_points, "trail_offset": trail_offset,
                 "armed": previous["armed"] if previous else False, "best": previous["best"] if previous else NAN,
                 "created": i}
        self.exits[key] = state
        return None

    def close(self, i, id_, qty_percent=NAN, immediately=False, when=True):
        if not self._active(i, when) or not self.pos:
            return None
        if str(id_) != self.entry_id:
            return None
        fraction = 1.0 if isna(qty_percent) else max(0.0, min(1.0, qty_percent / 100))
        self.market.append(("close", str(id_), 0, fraction))
        if self.fill_on_close or t_(immediately):
            self._fill_market(i, self.C[i])
        return None

    def close_all(self, i, immediately=False, when=True):
        if not self._active(i, when) or not self.pos:
            return None
        self.market.append(("close", None, 0, 1.0))
        if self.fill_on_close or t_(immediately):
            self._fill_market(i, self.C[i])
        return None

    def cancel(self, i, id_, when=True):
        if t_(when):
            self.entry_orders.pop(str(id_), None)
            for key in [k for k in self.exits if k[0] == str(id_)]:
                self.exits.pop(key, None)

    def allow_entry_in(self, direction):
        """``strategy.risk.allow_entry_in``: bare long eller bare short; motsatt inngang lukker i stedet."""
        text = str(direction)
        self.allowed = 1 if text.endswith(".long") else -1 if text.endswith(".short") else 0

    def cancel_all(self, i, when=True):
        if t_(when):
            self.entry_orders.clear()
            self.exits.clear()

    # ------------------------------------------------ bokføring

    def _open(self, k, side, price, id_):
        equity = self.cash
        if equity <= 0:
            return
        qty = equity / (price * (1 + self.cost))
        fee = qty * price * self.cost
        self.cash -= fee
        self.fees += fee
        self.pos, self.avg, self.entry_qty = side * qty, price, qty
        self.entry_bar, self.entry_id = k, id_
        self.trade_fees, self.trade_funding = fee, 0.0
        for state in self.exits.values():
            state["armed"], state["best"] = False, NAN

    def _close(self, k, price, fraction, reason):
        if not self.pos:
            return
        side = 1 if self.pos > 0 else -1
        qty = abs(self.pos) * fraction if fraction < 1 else abs(self.pos)
        qty = min(qty, abs(self.pos))
        gross = side * qty * (price - self.avg)
        fee = qty * price * self.cost
        self.cash += gross - fee
        self.fees += fee
        share = qty / abs(self.pos)
        trade_fees = self.trade_fees * share + fee
        trade_funding = self.trade_funding * share
        self.trade_fees -= self.trade_fees * share
        self.trade_funding -= trade_funding
        self.trades.append(Trade(side, self.entry_bar, self.avg, k, price, qty, gross - trade_fees - trade_funding,
                                 trade_fees, trade_funding, self.entry_id, reason))
        remaining = abs(self.pos) - qty
        if remaining <= abs(self.pos) * 1e-12:
            self.pos, self.avg, self.entry_qty, self.entry_id = 0.0, NAN, 0.0, ""
            # Exit-ordre lagt før forrige close hørte til posisjonen som nå er lukket.
            self.exits = {key: st for key, st in self.exits.items() if st["created"] >= k - 1}
        else:
            self.pos = side * remaining
        if self.cash <= 0:
            self.bankrupt = True

    def _fill_market(self, k, price):
        orders, self.market = self.market, []
        for kind, id_, side, fraction in orders:
            if self.bankrupt:
                return
            if kind == "close":
                self._close(k, price, fraction, "close")
                continue
            if (self.long_only or self.allowed > 0) and side < 0:
                if self.pos > 0:
                    self._close(k, price, 1.0, "short-signal (kun long)")
                continue
            if self.allowed < 0 and side > 0:
                if self.pos < 0:
                    self._close(k, price, 1.0, "long-signal (kun short)")
                continue
            if self.pos and (self.pos > 0) == (side > 0):
                continue
            if self.pos:
                self._close(k, price, 1.0, "snu")
            self._open(k, side, price, id_)

    # ------------------------------------------------ hvert lys

    def on_open(self, k):
        if k < self.first or self.bankrupt:
            return
        price = self.O[k]
        if self.funding is not None and self.pos:
            rate = self.funding[k]
            if rate:
                amount = self.pos * price * rate
                self.cash -= amount
                self.funding_paid += amount
                self.trade_funding += amount
        if self.market:
            self._fill_market(k, price)
        if self.entry_orders or (self.exits and self.pos):
            self._intrabar(k)

    def _levels(self):
        """Ventende stopp/limit-ordre som (side som handles, type, nivå, nøkkel)."""
        out = []
        if self.pos:
            side = 1 if self.pos > 0 else -1
            for key, st in self.exits.items():
                if (st["from"] and st["from"] != self.entry_id) or st["created"] < self.entry_bar - 1:
                    continue
                stop = st["stop"] if not isna(st["stop"]) else \
                    (self.avg - side * st["loss"] * self.mintick if not isna(st["loss"]) else NAN)
                limit = st["limit"] if not isna(st["limit"]) else \
                    (self.avg + side * st["profit"] * self.mintick if not isna(st["profit"]) else NAN)
                if not isna(stop):
                    out.append((-side, "stop", stop, ("exit", key)))
                if not isna(limit):
                    out.append((-side, "limit", limit, ("exit", key)))
                if st["armed"] and not isna(st["best"]) and not isna(st["trail_offset"]):
                    out.append((-side, "stop", st["best"] - side * st["trail_offset"] * self.mintick, ("trail", key)))
        for key, (side, limit, stop) in self.entry_orders.items():
            if self.pos and (self.pos > 0) == (side > 0):
                continue
            if self.long_only and side < 0:
                continue
            level, kind = (stop, "stop") if not isna(stop) else (limit, "limit")
            out.append((side, kind, level, ("entry", key)))
        return out

    def _trail_update(self, price):
        if not self.pos:
            return
        side = 1 if self.pos > 0 else -1
        for st in self.exits.values():
            if isna(st["trail_offset"]):
                continue
            activation = st["trail_price"] if not isna(st["trail_price"]) else \
                (self.avg + side * st["trail_points"] * self.mintick if not isna(st["trail_points"]) else NAN)
            if isna(activation):
                continue
            if not st["armed"] and side * (price - activation) >= 0:
                st["armed"], st["best"] = True, price
            elif st["armed"]:
                st["best"] = max(st["best"], price) if side > 0 else min(st["best"], price)

    def _intrabar(self, k):
        o, h, l, c = self.O[k], self.H[k], self.L[k], self.C[k]
        path = [o, h, l, c] if (h - o) < (o - l) else [o, l, h, c]
        for step in range(3):
            a, b = path[step], path[step + 1]
            for _ in range(4):        # flere ordre kan fylles etter hverandre i samme bevegelse
                hit = self._first_hit(a, b, gap=step == 0)
                if hit is None:
                    break
                side, kind, level, key, price = hit
                if key[0] == "entry":
                    self.entry_orders.pop(key[1], None)
                    self.market.append(("entry", key[1], side, NAN))
                    self._fill_market(k, price)
                else:
                    state = self.exits.get(key[1])
                    if state is None:
                        continue
                    fraction = min(1.0, self.entry_qty * state["qp"] / 100 / abs(self.pos)) if self.pos else 1.0
                    self.exits.pop(key[1], None)
                    reason = {"stop": "stopp", "limit": "mål"}[kind] if key[0] == "exit" else "trailing-stopp"
                    self._close(k, price, fraction, reason)
                a = price
                if self.bankrupt:
                    return
            self._trail_update(b)

    def _first_hit(self, a, b, gap):
        best = None
        up = b >= a
        for side, kind, level, key in self._levels():
            if isna(level):
                continue
            rising_trigger = (side > 0 and kind == "stop") or (side < 0 and kind == "limit")
            if gap and ((rising_trigger and a >= level) or (not rising_trigger and a <= level)):
                return side, kind, level, key, a
            if rising_trigger and up and a <= level <= b:
                distance = level - a
            elif not rising_trigger and not up and b <= level <= a:
                distance = a - level
            else:
                continue
            if best is None or distance < best[0]:
                best = (distance, (side, kind, level, key, level))
        return None if best is None else best[1]

    def on_close(self, k):
        if k < self.first:
            return
        if self.bankrupt:
            self.equity[k] = 0.0
            return
        value = self.equity_now(k)
        if value <= 0:
            self.bankrupt, value = True, 0.0
            self.pos = 0.0
        self.equity[k] = value
        if self.pos:
            self.exposed_bars += 1

    def finish(self, k):
        """Lukk åpen posisjon ved siste close, så sluttresultatet er realisert og sammenlignbart."""
        if self.pos and not self.bankrupt:
            self._close(k, self.C[k], 1.0, "åpen ved slutt")
            self.equity[k] = self.cash
