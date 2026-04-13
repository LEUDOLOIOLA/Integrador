"""
main.py — Entry-point for UTR221 ML on the Siemens IOT2050.

Orchestrates the full pipeline:
  1. Load and merge source data (urt220 + urt221)
  2. Build features (lags, trends, rolling mean, hour-of-day)
  3. Train RandomForestRegressor and evaluate on the held-out test set
  4. Run a recursive 7-hour ahead forecast with pump-state simulation
  5. Export forecast to Excel and optionally save a PNG chart

Logging is written to both stdout and a rotating log file so that
unattended gateway operation leaves a permanent audit trail.

Exit code 1 is returned on any unhandled error so that systemd or
another process supervisor can detect and restart the service.
"""

import logging
import logging.handlers
import sys

from config import Config
from data_loader import load_data
from feature_engineering import build_features, split_dataframes
from forecasting import run_forecast
from model import evaluate_model, log_feature_importance, train_model
from reporting import export_chart, export_excel


def setup_logging(cfg: Config) -> None:
    """Configure root logger with a console handler and a rotating file handler."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            cfg.log_file,
            maxBytes=1_000_000,   # 1 MB per file
            backupCount=2,
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def main() -> None:
    cfg = Config()
    setup_logging(cfg)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  UTR221 ML — Previsão de Pressão (IOT2050)")
    logger.info("=" * 60)

    try:
        # 1. Load data
        df_raw = load_data(cfg)

        # 2. Feature engineering
        df = build_features(df_raw, cfg)
        df, df2, _ = split_dataframes(df, cfg)

        # 3. Train & evaluate
        modelo, _X_test, y_test, previsoes = train_model(df2, cfg)
        log_feature_importance(modelo, cfg)
        evaluate_model(y_test, previsoes)

        # 4. Recursive forecast
        # df (feature-engineered) seeds the forecast with the last known
        # pressure history, pump states and UTR-220 level.
        df_futuro = run_forecast(df, modelo, cfg)
        logger.info(
            "Primeiros 10 passos:\n%s",
            df_futuro[["Previsto_Futuro", "Hora"]].head(10).to_string(),
        )
        logger.info(
            "Últimos 10 passos:\n%s",
            df_futuro[["Previsto_Futuro", "Hora"]].tail(10).to_string(),
        )

        # 5. Export
        export_excel(df_futuro, cfg)
        export_chart(df, df_futuro, cfg)

        logger.info("=" * 60)
        logger.info("  Processamento concluído com sucesso.")
        logger.info("=" * 60)

    except Exception as exc:
        logger.critical("Erro fatal: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
