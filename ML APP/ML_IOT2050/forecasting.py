"""
forecasting.py — Recursive pressure forecasting for UTR221 ML (IOT2050).

The PumpController class encapsulates the hysteresis (bang-bang) pump
control logic so it can be tuned or extended independently of the
forecasting loop (e.g. to add pump-rotation scheduling).

The run_forecast function performs the multi-step ahead prediction,
feeding each predicted value back as a feature for the next step.
A step-level try/except ensures a single bad prediction does not abort
the entire forecast run — the last valid value is reused instead.
"""

import logging

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


class PumpController:
    """
    Bang-bang (hysteresis) controller for UTR-220 feed pumps.

    Rules
    -----
    nivel >= setpoint_alto  → turn both pumps OFF  (0, 0)
    nivel <= setpoint_baixo → turn both pumps ON   (1, 1)
    otherwise               → keep current state
    """

    def __init__(self, cfg: Config) -> None:
        self.setpoint_alto  = cfg.setpoint_alto
        self.setpoint_baixo = cfg.setpoint_baixo

    def update(self, nivel: float, bomba1: float, bomba2: float):
        """Return the new (bomba1, bomba2) state for the given pressure level."""
        if nivel >= self.setpoint_alto:
            return 0.0, 0.0
        if nivel <= self.setpoint_baixo:
            return 1.0, 1.0
        return bomba1, bomba2


def _build_step_features(
    bomba1: float,
    bomba2: float,
    nivel220_const: float,
    nivel_lag1: float,
    nivel_lag2: float,
    nivel_lag60: float,
    proximo_ts: pd.Timestamp,
    historico: list,
) -> dict:
    """Assemble the feature dict for a single forecast step."""
    return {
        "BOMBA_1":        bomba1,
        "BOMBA_2":        bomba2,
        "nivel_220_lag1": nivel220_const,
        "nivel_221_lag1": nivel_lag1,
        "hora":           proximo_ts.hour,
        "tendencia_1min": nivel_lag1 - nivel_lag2,
        "tendencia_1h":   nivel_lag1 - nivel_lag60,
        "media_movel_1h": float(np.mean(historico[-60:])),
    }


def run_forecast(df: pd.DataFrame, modelo, cfg: Config) -> pd.DataFrame:
    """
    Execute a recursive cfg.forecast_steps-minute ahead forecast.

    Parameters
    ----------
    df     : feature-engineered DataFrame (must contain pressure, pump and
             level columns; its last row seeds the forecast)
    modelo : fitted sklearn estimator exposing .predict()
    cfg    : Config instance

    Returns
    -------
    DataFrame indexed by forecast timestamp with columns:
    Previsto_Futuro, BOMBA_1, BOMBA_2, Passo, Hora.
    """
    if len(df) < 2:
        raise ValueError(
            "DataFrame insuficiente para iniciar previsão (mínimo 2 linhas)."
        )

    ultimo_ts      = df.index[-1]
    historico      = list(df[cfg.col_pressao221].tail(60).values)
    bomba1         = float(df[cfg.col_bomba1].iloc[-1])
    bomba2         = float(df[cfg.col_bomba2].iloc[-1])
    nivel220_const = float(df[cfg.col_nivel220].iloc[-1])

    controller = PumpController(cfg)
    registros: list = []

    logger.info("Último dado real: %s", ultimo_ts)
    logger.info(
        "Iniciando previsão recursiva: %d passos (%dh) ...",
        cfg.forecast_steps,
        cfg.forecast_steps // 60,
    )

    for passo in range(1, cfg.forecast_steps + 1):
        proximo_ts = ultimo_ts + pd.Timedelta(minutes=passo)

        nivel_lag1  = historico[-1]
        nivel_lag2  = historico[-2]
        nivel_lag60 = historico[-60] if len(historico) >= 60 else historico[0]

        bomba1, bomba2 = controller.update(nivel_lag1, bomba1, bomba2)

        feat_values = _build_step_features(
            bomba1, bomba2, nivel220_const,
            nivel_lag1, nivel_lag2, nivel_lag60,
            proximo_ts, historico,
        )

        try:
            X_futuro = pd.DataFrame([feat_values])[cfg.features]
            previsao = max(float(modelo.predict(X_futuro)[0]), 0.0)
        except Exception as exc:
            logger.warning(
                "Passo %d: erro na predição (%s). Usando último valor.", passo, exc
            )
            previsao = nivel_lag1

        registros.append({
            "Data":            proximo_ts,
            "Previsto_Futuro": round(previsao, 2),
            "BOMBA_1":         bomba1,
            "BOMBA_2":         bomba2,
            "Passo":           passo,
            "Hora":            proximo_ts.strftime("%H:%M"),
        })

        historico.append(previsao)

    df_futuro = pd.DataFrame(registros).set_index("Data")
    logger.info("Previsão gerada: %d passos.", len(df_futuro))
    return df_futuro
