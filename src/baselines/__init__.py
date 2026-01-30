"""Baselines module."""
from src.baselines.routers import (
    AlwaysDirectRouter, AlwaysCoTRouter, RandomRouter,
    LengthRouter, KeywordRouter, EntropyRouter,
    SelfConsistencyRouter, OracleRouter, EmbeddingRouter,
    get_all_baselines
)
