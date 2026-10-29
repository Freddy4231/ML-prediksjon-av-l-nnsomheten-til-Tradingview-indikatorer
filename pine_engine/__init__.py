"""Pine Script-tolk for lønnsomhetstesten på undersiden «Pine-test».

Leser TradingView-kode (Pine v4–v6), oversetter den til begrenset Python og kjører den lys for
lys på ekte markedsdata. Brukerkoden kjøres aldri direkte: bare syntakstreet oversettes, alle
navn får prefiks, og bare funksjoner fra tolkens egen liste kan kalles.
"""
from pine_engine.parser import PineError, parse  # noqa: F401
