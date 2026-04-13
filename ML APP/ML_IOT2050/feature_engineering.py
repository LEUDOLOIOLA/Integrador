"""
feature_engineering.py — Feature construction for UTR221 ML (IOT2050).

Builds lag variables, trend indicators, rolling mean and the hour-of-day
feature required by the RandomForest model.  All operations are applied
to a copy of the input DataFrame so the caller's data is never mutated.
"""

import logging

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Add derived feature columns and drop rows with NaN values.

    Features added
    --------------
    target          : pressure 1 minute ahead (prediction target)
    nivel_221_lag1  : pressure 1 minute ago
    nivel_220_lag1  : UTR-220 level 1 minute ago
    tendencia_1min  : pressure change over the last minute
    tendencia_1h    : pressure change over the last hour (60 min)
    media_movel_1h  : rolling mean pressure over the last hour
    hora            : hour-of-day extracted from the timestamp index
    """
    col = cfg.col_pressao221
    df = df.copy()

    df["target"]         = df[col].shift(-1)
    df["nivel_221_lag1"] = df[col].shift(1)
    df["nivel_220_lag1"] = df[cfg.col_nivel220].shift(1)
    df["tendencia_1min"] = df[col].shift(1) - df[col].shift(2)
    df["tendencia_1h"]   = df[col].shift(1) - df[col].shift(60)
    df["media_movel_1h"] = df[col].shift(1).rolling(window=60).mean()
    df["hora"]           = df.index.hour

    df = df.dropna()
    logger.info("Features construídas: %d linhas válidas.", len(df))
    return df


def split_dataframes(df: pd.DataFrame, cfg: Config):
    """
    Partition df into three subsets:

    df  — complete feature DataFrame
    df2 — rows where both target and nivel_221_lag1 are non-zero
          (used for training to exclude pump-off / zero-pressure periods)
    df3 — rows where the raw pressure reading equals zero
          (diagnostic subset)

    Returns (df, df2, df3).
    """
    df2 = df[(df["target"] != 0) & (df["nivel_221_lag1"] != 0)].copy()
    df3 = df[df[cfg.col_pressao221] == 0].copy()

    logger.info(
        "DF (original): %d | DF2 (sem zeros): %d (%d removidas) | DF3 (PRESSAO=0): %d",
        len(df),
        len(df2),
        len(df) - len(df2),
        len(df3),
    )
    return df, df2, df3
