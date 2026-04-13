"""
config.py — Centralised configuration for UTR221 ML (IOT2050).

All paths, column names, model hyper-parameters and operational
setpoints are defined here so that operator adjustments never
require touching the processing modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Directory that contains the source Excel files (ML APP/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    file_urt220: Path = field(default_factory=lambda: BASE_DIR / "urt220.xlsx")
    file_urt221: Path = field(default_factory=lambda: BASE_DIR / "urt221.xlsx")
    output_excel: Path = field(default_factory=lambda: BASE_DIR / "DF6_futuro_7h.xlsx")
    output_chart: Path = field(default_factory=lambda: BASE_DIR / "previsao_futura_7h.png")
    log_file: Path = field(default_factory=lambda: BASE_DIR / "utr221_ml.log")

    # --------------------------------------------------- raw column names
    col_timestamp: str = "E3TIMESTAMP"
    col_bomba1_raw: str = "CMB_220_S2A_EST"
    col_bomba2_raw: str = "CMB_220_S2B_EST"
    col_nivel220_raw: str = "LIT_220_RA2_000"
    col_pressao221_raw: str = "PIT_221_S01_000"

    # -------------------------------------------- renamed column names
    col_bomba1: str = "BOMBA_1"
    col_bomba2: str = "BOMBA_2"
    col_nivel220: str = "NIVEL-UTR-220"
    col_pressao221: str = "PRESSAO-UTR-221"

    # -------------------------------------------------- timestamp format
    ts_format: str = "%d/%m/%y %H:%M:%S.%f"

    # ------------------------------------------------------- resampling
    resample_freq: str = "1min"

    # ---------------------------------------------- data filter start
    filter_start: str = "2026-02-28 17:30:38"

    # --------------------------------------------------------- features
    features: List[str] = field(default_factory=lambda: [
        "BOMBA_1",
        "BOMBA_2",
        "nivel_220_lag1",
        "nivel_221_lag1",
        "hora",
        "tendencia_1min",
        "tendencia_1h",
        "media_movel_1h",
    ])

    # ------------------------------------------------------- training
    train_split_ratio: float = 0.8
    n_estimators: int = 100
    random_state: int = 42

    # ----------------------------------------------------- forecasting
    forecast_steps: int = 420       # 7 hours × 60 minutes
    setpoint_alto: float = 1.68     # pressure level to turn pumps OFF
    setpoint_baixo: float = 1.55    # pressure level to turn pumps ON

    # ------------------------------------------------------- reporting
    generate_chart: bool = True
    chart_dpi: int = 100            # lower than 300 to save memory/disk
