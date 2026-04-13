"""
model.py — Model training and evaluation for UTR221 ML (IOT2050).

Uses RandomForestRegressor with a temporal (no-shuffle) train/test split.
n_jobs is fixed at 1 to avoid spawning extra threads that would compete
for the IOT2050's dual-core CPU during deployment.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config

logger = logging.getLogger(__name__)


def train_model(df2: pd.DataFrame, cfg: Config):
    """
    Perform a temporal 80/20 train/test split on df2 and fit the model.

    Returns
    -------
    modelo     : fitted RandomForestRegressor
    X_test     : test feature DataFrame
    y_test     : test target Series
    previsoes  : ndarray of predictions on X_test
    """
    X = df2[cfg.features]
    y = df2["target"]

    split = int(len(df2) * cfg.train_split_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(
        "Treinando RandomForestRegressor (n_estimators=%d) com %d amostras ...",
        cfg.n_estimators,
        len(X_train),
    )
    try:
        modelo = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=1,
        )
        modelo.fit(X_train, y_train)
    except MemoryError:
        logger.critical(
            "Memória insuficiente para treinar o modelo. "
            "Reduza n_estimators em config.py."
        )
        raise

    previsoes = modelo.predict(X_test)
    logger.info("Treinamento concluído.")
    return modelo, X_test, y_test, previsoes


def log_feature_importance(modelo, cfg: Config) -> None:
    """Log a ranked bar chart of feature importances using block characters."""
    importancias = (
        pd.Series(modelo.feature_importances_, index=cfg.features)
        .sort_values(ascending=False)
    )
    logger.info("=" * 40)
    logger.info("   COEFICIENTES (Importância)")
    logger.info("=" * 40)
    for feat, imp in importancias.items():
        barra = "█" * int(imp * 40)
        logger.info("%-20s %.4f  %s", feat, imp, barra)
    logger.info("=" * 40)


def evaluate_model(y_test: pd.Series, previsoes: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, R², MAPE and Accuracy.

    Returns a dict with keys: mae, rmse, r2, mape, acuracia.
    """
    mae  = float(mean_absolute_error(y_test, previsoes))
    rmse = float(np.sqrt(mean_squared_error(y_test, previsoes)))
    r2   = float(r2_score(y_test, previsoes))

    mask = y_test != 0
    mape     = float(np.mean(np.abs((y_test[mask] - previsoes[mask]) / y_test[mask])) * 100)
    acuracia = 100.0 - mape

    metrics = dict(mae=mae, rmse=rmse, r2=r2, mape=mape, acuracia=acuracia)

    logger.info("=" * 40)
    logger.info("   MÉTRICAS DO MODELO")
    logger.info("=" * 40)
    logger.info("MAE  (Erro Médio Absoluto):  %.4f", mae)
    logger.info("RMSE (Raiz do Erro Quadr.):  %.4f", rmse)
    logger.info("R²   (Coef. Determinação):   %.4f", r2)
    logger.info("MAPE (Erro %% Médio Abs.):    %.2f%%", mape)
    logger.info("Acurácia (100 - MAPE):       %.2f%%", acuracia)
    logger.info("=" * 40)

    return metrics
