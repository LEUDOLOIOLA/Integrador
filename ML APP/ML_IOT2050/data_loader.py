"""
data_loader.py — File I/O and timestamp parsing for UTR221 ML (IOT2050).

Responsibilities:
- Load urt220.xlsx and urt221.xlsx with column validation.
- Parse the E3TIMESTAMP format used by the E3 SCADA system.
- Resample to 1-minute intervals, merge and apply the date filter.
- Cast numeric columns to float32 to reduce memory usage on the IOT2050.
"""

import logging
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def parse_timestamp(series: pd.Series, fmt: str) -> pd.Series:
    """
    Parse E3TIMESTAMP strings into datetime objects.

    The SCADA export uses a comma as the decimal separator for the
    fractional seconds and may include more than 6 decimal digits,
    e.g. '01/02/26 00:00:27,954000000'.

    Steps:
      1. Replace ',' with '.' in the fractional part.
      2. Truncate to exactly 6 decimal digits (microseconds).
      3. Parse with the given strftime format.
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
    )
    return pd.to_datetime(cleaned, format=fmt)


def _load_excel(path: Path, usecols: list, ts_col: str, fmt: str) -> pd.DataFrame:
    """Load a single Excel file with validation and memory optimisation."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    logger.info("Carregando %s ...", path.name)
    try:
        df = pd.read_excel(path, usecols=usecols, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Erro ao ler '{path.name}': {exc}") from exc

    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em '{path.name}': {missing}")

    df[ts_col] = parse_timestamp(df[ts_col], fmt)
    df = df.set_index(ts_col).sort_index()

    # Reduce memory: downcast float64 → float32
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    return df


def load_data(cfg: Config) -> pd.DataFrame:
    """
    Load, resample, merge and filter both source files.

    Returns a merged DataFrame indexed by minute-frequency timestamps
    from cfg.filter_start onwards.
    """
    cols220 = [cfg.col_timestamp, cfg.col_bomba1_raw, cfg.col_bomba2_raw, cfg.col_nivel220_raw]
    cols221 = [cfg.col_timestamp, cfg.col_pressao221_raw]

    df220 = _load_excel(cfg.file_urt220, cols220, cfg.col_timestamp, cfg.ts_format)
    df221 = _load_excel(cfg.file_urt221, cols221, cfg.col_timestamp, cfg.ts_format)

    df220 = df220.resample(cfg.resample_freq).mean().dropna()
    df221 = df221.resample(cfg.resample_freq).mean().dropna()

    df = df220.join(df221, how="inner")
    df = df.loc[cfg.filter_start:]

    df = df.rename(columns={
        cfg.col_bomba1_raw:    cfg.col_bomba1,
        cfg.col_bomba2_raw:    cfg.col_bomba2,
        cfg.col_nivel220_raw:  cfg.col_nivel220,
        cfg.col_pressao221_raw: cfg.col_pressao221,
    })

    logger.info("Dados carregados: %d linhas após filtragem.", len(df))
    return df
