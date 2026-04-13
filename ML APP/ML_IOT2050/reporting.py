"""
reporting.py — Output generation for UTR221 ML (IOT2050).

Exports the forecast DataFrame to Excel and optionally produces a PNG
chart showing the last hour of real data alongside the 7-hour forecast.

Chart generation can be disabled via cfg.generate_chart = False for
headless or low-resource deployments.  The Agg backend is selected
explicitly so that no display server is required.
"""

import logging

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def export_excel(df_futuro: pd.DataFrame, cfg: Config) -> None:
    """Save the forecast DataFrame to the path defined in cfg.output_excel."""
    try:
        df_futuro.to_excel(cfg.output_excel, engine="openpyxl")
        logger.info("✅ Arquivo gerado: %s", cfg.output_excel.name)
    except (PermissionError, OSError) as exc:
        logger.error("Não foi possível salvar '%s': %s", cfg.output_excel.name, exc)


def export_chart(df: pd.DataFrame, df_futuro: pd.DataFrame, cfg: Config) -> None:
    """
    Render and save a PNG comparing the last hour of real pressure data
    with the full forecast horizon.

    The chart is skipped silently when cfg.generate_chart is False.
    Memory is released via plt.close('all') in a finally block.
    """
    if not cfg.generate_chart:
        logger.info("Geração de gráfico desabilitada (generate_chart=False).")
        return

    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — no display needed
        import matplotlib.pyplot as plt  # noqa: PLC0415

        ultimo_ts     = df.index[-1]
        janela_inicio = ultimo_ts - pd.Timedelta(hours=1)
        dados_grafico = df.loc[janela_inicio:ultimo_ts]

        plt.figure(figsize=(15, 8))
        plt.plot(
            dados_grafico.index,
            dados_grafico[cfg.col_pressao221],
            label="Última 1h Real",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            df_futuro.index,
            df_futuro["Previsto_Futuro"],
            label="Previsão 7h Futuras",
            color="green",
            linewidth=2,
            linestyle="--",
        )
        plt.axvline(
            ultimo_ts,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="Limite dos dados reais",
        )

        plt.title(f"Previsão Futura 7 Horas\nA partir de {ultimo_ts}", fontsize=14)
        plt.xlabel("Timestamp")
        plt.ylabel("Pressão (BAR)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(cfg.output_chart, dpi=cfg.chart_dpi, bbox_inches="tight")
        logger.info("✅ Gráfico salvo: %s", cfg.output_chart.name)

    except Exception as exc:
        logger.error("Erro ao gerar gráfico: %s", exc)
    finally:
        if plt is not None:
            plt.close("all")
