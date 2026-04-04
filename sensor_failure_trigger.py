"""
Gatilho de Falha de Sensor - UTR 221
=====================================
Detecta automaticamente falhas no sensor NIVEL-UTR-221 (leituras zero ou
ausentes) e dispara a inferência autorregressiva para prever o nível do
reservatório nas próximas 4 janelas de 15 minutos (1 hora).

Uso:
    python sensor_failure_trigger.py

O script carrega os dados, detecta o primeiro evento de falha, treina o
melhor modelo e gera a previsão a partir do instante da falha.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# CONSTANTES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO = os.path.join(BASE_DIR, 'UTR221_2025_2026_15MIN.xls')
ARQUIVO_SAIDA = os.path.join(BASE_DIR, 'previsao_falha_sensor.xlsx')

TARGET = 'NIVEL-UTR-221'
DIAS_SEMANA_COLS = [f'DiaSemana_{i}' for i in range(7)]
FEATURES = [
    'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
    'NIVEL-UTR-220',
    'Hora', 'Minuto', 'Dia', 'Periodo',
] + DIAS_SEMANA_COLS + [
    'NIVEL221_lag1', 'NIVEL221_lag2', 'NIVEL221_lag4',
    'NIVEL220_lag1',
    'Tendencia_Nivel221',
    'Tendencia_Nivel220',
]

# Número de passos futuros (4 × 15 min = 1 hora)
PASSOS = 4

# Mínimo de registros necessários para treinar o modelo
MIN_TRAINING_SAMPLES = 20


# =============================================================================
# 1. DETECÇÃO DE FALHA DE SENSOR
# =============================================================================
def detect_sensor_failure(df, col=TARGET, consecutive=1):
    """Detecta o primeiro evento de falha no sensor.

    Uma falha é definida como ``consecutive`` ou mais leituras consecutivas
    iguais a zero (ou NaN) na coluna *col*.  Retorna o timestamp da primeira
    leitura nula/zero da primeira sequência que satisfaz o critério, ou
    ``None`` se nenhuma falha for encontrada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna *col* e coluna 'Data'.
    col : str
        Nome da coluna de nível a monitorar.
    consecutive : int
        Número mínimo de leituras consecutivas problemáticas para declarar
        falha.  O padrão é 1 (qualquer leitura zero/NaN dispara a falha).

    Returns
    -------
    pd.Timestamp or None
        Instante da primeira leitura problemática da primeira sequência de
        falha, ou ``None`` se não houver falha.
    """
    problema = (df[col].isna()) | (df[col] == 0)
    contagem = 0
    inicio_falha = None

    for idx in df.index:
        if problema[idx]:
            contagem += 1
            if contagem == 1:
                inicio_falha = df.at[idx, 'Data']
            if contagem >= consecutive:
                return inicio_falha
        else:
            contagem = 0
            inicio_falha = None

    return None


# =============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================
def prepare_data(arquivo=ARQUIVO):
    """Carrega, limpa e engenharia de features do arquivo XLS.

    Returns
    -------
    pd.DataFrame
        DataFrame pronto para treinamento, com todas as features calculadas.
    """
    df = pd.read_excel(arquivo, header=None, skiprows=2)
    if df.shape[1] == 6:
        df.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                      'NIVEL-UTR-220', 'NIVEL-UTR-221', 'Obs']
        df = df.drop('Obs', axis=1)
    elif df.shape[1] == 5:
        df.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                      'NIVEL-UTR-220', 'NIVEL-UTR-221']
    else:
        raise ValueError(
            f'Formato inesperado em {os.path.basename(arquivo)}: '
            f'{df.shape[1]} colunas.'
        )

    colunas_numericas = [
        'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
        'NIVEL-UTR-220', 'NIVEL-UTR-221',
    ]
    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Data'] + colunas_numericas).reset_index(drop=True)
    df = df.sort_values('Data').reset_index(drop=True)

    # Imputação de zeros usando tendência histórica
    for col in ['NIVEL-UTR-220', 'NIVEL-UTR-221']:
        lag1 = df[col].shift(1)
        lag2 = df[col].shift(2)
        mask = (lag1 > 0) & (lag2 > 0)
        media_tend = (lag1 - lag2)[mask].mean()
        zeros_idx = df[df[col] == 0].index
        for idx in zeros_idx:
            if idx == 0:
                prox = df.loc[df.index > idx, col]
                prox_valido = prox[prox > 0]
                fallback_value = df.loc[df[col] > 0, col].mean()
                df.at[idx, col] = prox_valido.iloc[0] if not prox_valido.empty else fallback_value
            else:
                df.at[idx, col] = df.at[idx - 1, col] + media_tend

    # Engenharia de features temporais
    df['Hora'] = df['Data'].dt.hour
    df['Minuto'] = df['Data'].dt.minute
    _dias = pd.get_dummies(df['Data'].dt.dayofweek, prefix='DiaSemana').astype(int)
    for c in DIAS_SEMANA_COLS:
        if c not in _dias.columns:
            _dias[c] = 0
    _dias = _dias[DIAS_SEMANA_COLS]
    df = pd.concat([df, _dias], axis=1)
    df['Dia'] = df['Data'].dt.day
    df['Periodo'] = df['Hora'].apply(
        lambda h: 0 if h < 6 else (1 if h < 12 else (2 if h < 18 else 3))
    )

    # Features de lag
    df['NIVEL221_lag1'] = df['NIVEL-UTR-221'].shift(1)
    df['NIVEL221_lag2'] = df['NIVEL-UTR-221'].shift(2)
    df['NIVEL221_lag4'] = df['NIVEL-UTR-221'].shift(4)
    df['NIVEL220_lag1'] = df['NIVEL-UTR-220'].shift(1)
    df['Tendencia_Nivel221'] = df['NIVEL221_lag1'] - df['NIVEL221_lag2']
    df['Tendencia_Nivel220'] = df['NIVEL-UTR-220'].shift(1) - df['NIVEL-UTR-220'].shift(2)

    df = df.dropna().reset_index(drop=True)
    return df


# =============================================================================
# 3. TREINAMENTO DO MELHOR MODELO
# =============================================================================
def train_best_model(df):
    """Treina múltiplos modelos e retorna o melhor (maior R²) junto com o
    scaler e um flag indicando se o scaler deve ser usado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com features e target já calculados.

    Returns
    -------
    tuple : (model, scaler, use_scaler, model_name)
    """
    X = df[FEATURES]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42,
            n_jobs=max(1, (os.cpu_count() or 1) - 1),
        ),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    lineares = {'Linear Regression', 'Ridge Regression'}

    resultados = {}
    for nome, modelo in modelos.items():
        if nome in lineares:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        resultados[nome] = {'MAE': mae, 'R²': r2, 'model': modelo}
        print(f'  {nome:>25s} | MAE={mae:.4f} m | R²={r2:.4f}')

    melhor_nome = max(resultados, key=lambda k: resultados[k]['R²'])
    melhor_modelo = resultados[melhor_nome]['model']
    usar_scaler = melhor_nome in lineares
    print(f'\n  Melhor modelo: {melhor_nome} (R²={resultados[melhor_nome]["R²"]:.4f})')
    return melhor_modelo, scaler, usar_scaler, melhor_nome


# =============================================================================
# 4. INFERÊNCIA AUTORREGRESSIVA A PARTIR DA FALHA
# =============================================================================
def run_autoregressive_forecast(df, model, scaler, use_scaler, failure_time, steps=PASSOS):
    """Executa inferência autorregressiva a partir do instante de falha.

    Usa dados históricos reais até *failure_time* e, a cada passo, alimenta
    a previsão anterior como entrada do próximo passo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo com features (usado para lags e dados reais de
        NIVEL-UTR-220 quando disponíveis).
    model : estimator
        Modelo treinado (scikit-learn).
    scaler : StandardScaler
        Scaler ajustado nos dados de treino.
    use_scaler : bool
        Se ``True``, aplica o scaler na entrada antes da predição.
    failure_time : pd.Timestamp
        Instante da falha (último timestamp com dado real).
    steps : int
        Número de passos futuros a prever.

    Returns
    -------
    pd.DataFrame
        Tabela com colunas 'Data', 'Previsto (m)', 'Real (m)', 'Erro (m)'.
    """
    hist = df[df['Data'] <= failure_time].copy()
    if len(hist) < 5:
        raise ValueError(
            f'Dados insuficientes antes de {failure_time} para inferência. '
            f'Necessário pelo menos 5 registros, encontrados: {len(hist)}.'
        )

    niveis_conhecidos = list(hist['NIVEL-UTR-221'].iloc[-4:].values)
    prev_nivel220 = hist['NIVEL-UTR-220'].iloc[-1]
    previsoes_futuras = []

    print(f'\n  Falha detectada em: {failure_time}')
    print(f'  Último nível real: {niveis_conhecidos[-1]:.2f} m')
    print(f'  {"Instante":<22} {"Previsto (m)":>14} {"Real (m)":>10}')
    print('  ' + '-' * 50)

    for passo in range(1, steps + 1):
        dt = failure_time + pd.Timedelta(minutes=15 * passo)

        # Dados reais de CMB e NIVEL-220 quando disponíveis no dataset
        linha_real = df[df['Data'] == dt]
        if not linha_real.empty:
            cmb01 = int(linha_real['CMB01-UTR-220-ESTADO'].values[0])
            cmb02 = int(linha_real['CMB02-UTR-220-ESTADO'].values[0])
            nivel220 = linha_real['NIVEL-UTR-220'].values[0]
        else:
            cmb01 = int(hist['CMB01-UTR-220-ESTADO'].iloc[-1])
            cmb02 = int(hist['CMB02-UTR-220-ESTADO'].iloc[-1])
            nivel220 = prev_nivel220

        # Tendência de NIVEL-220
        nivel220_ant1 = prev_nivel220
        lr_ant2 = df[df['Data'] == (dt - pd.Timedelta(minutes=30))]
        nivel220_ant2 = (
            lr_ant2['NIVEL-UTR-220'].values[0] if not lr_ant2.empty else nivel220_ant1
        )

        # Lags de NIVEL-221 usando previsões acumuladas
        todos = niveis_conhecidos + previsoes_futuras
        lag1 = todos[-1]
        lag2 = todos[-2]
        lag4 = todos[-4] if len(todos) >= 4 else niveis_conhecidos[0]

        registro = {
            'CMB01-UTR-220-ESTADO': cmb01,
            'CMB02-UTR-220-ESTADO': cmb02,
            'NIVEL-UTR-220': nivel220,
            'Hora': dt.hour,
            'Minuto': dt.minute,
            'Dia': dt.day,
            **{c: int(dt.dayofweek == i) for i, c in enumerate(DIAS_SEMANA_COLS)},
            'Periodo': 0 if dt.hour < 6 else (
                1 if dt.hour < 12 else (2 if dt.hour < 18 else 3)),
            'NIVEL221_lag1': lag1,
            'NIVEL221_lag2': lag2,
            'NIVEL221_lag4': lag4,
            'NIVEL220_lag1': nivel220_ant1,
            'Tendencia_Nivel221': lag1 - lag2,
            'Tendencia_Nivel220': nivel220_ant1 - nivel220_ant2,
        }

        entrada = pd.DataFrame([registro])[FEATURES]
        if use_scaler:
            pred = model.predict(scaler.transform(entrada))[0]
        else:
            pred = model.predict(entrada)[0]

        previsoes_futuras.append(pred)
        prev_nivel220 = nivel220

        real_val = linha_real['NIVEL-UTR-221'].values[0] if not linha_real.empty else np.nan
        real_str = f'{real_val:.2f}' if not np.isnan(real_val) else '—'
        print(f'  {str(dt):<22} {pred:>14.2f} {real_str:>10}')

    # Montar tabela de resultados
    datas = [failure_time + pd.Timedelta(minutes=15 * i) for i in range(1, steps + 1)]
    reais = []
    for dt in datas:
        lr = df[df['Data'] == dt]
        reais.append(lr['NIVEL-UTR-221'].values[0] if not lr.empty else np.nan)

    tabela = pd.DataFrame({
        'Data': datas,
        'Previsto (m)': previsoes_futuras,
        'Real (m)': reais,
    })
    tabela['Erro (m)'] = tabela['Real (m)'] - tabela['Previsto (m)']
    return tabela


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================
def main(arquivo=ARQUIVO, arquivo_saida=ARQUIVO_SAIDA, consecutive=1):
    """Pipeline completo: detectar falha → treinar modelo → prever → exportar.

    Parameters
    ----------
    arquivo : str
        Caminho para o arquivo XLS de entrada.
    arquivo_saida : str
        Caminho para o arquivo XLSX de saída com as previsões.
    consecutive : int
        Número mínimo de leituras consecutivas com zero/NaN para declarar
        falha de sensor (padrão: 1).
    """
    print('=' * 60)
    print('GATILHO DE FALHA DE SENSOR — UTR 221')
    print('=' * 60)

    # 1. Carregar e preparar dados
    print('\n[1/4] Carregando e preparando dados...')
    df = prepare_data(arquivo)
    print(f'      {len(df)} registros carregados.')

    # 2. Detectar falha de sensor nos dados brutos (antes da imputação de zeros)
    #    Para isso, recarregamos os dados sem a imputação a fim de identificar
    #    as leituras originalmente zeradas.
    print('\n[2/4] Detectando falha de sensor...')
    df_raw = pd.read_excel(arquivo, header=None, skiprows=2)
    if df_raw.shape[1] == 6:
        df_raw.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                          'NIVEL-UTR-220', 'NIVEL-UTR-221', 'Obs']
    elif df_raw.shape[1] == 5:
        df_raw.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                          'NIVEL-UTR-220', 'NIVEL-UTR-221']
    df_raw['Data'] = pd.to_datetime(df_raw['Data'], format='%d/%m/%Y %H:%M', errors='coerce')
    df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors='coerce')
    df_raw = df_raw.dropna(subset=['Data', TARGET]).sort_values('Data').reset_index(drop=True)

    failure_time = detect_sensor_failure(df_raw, col=TARGET, consecutive=consecutive)

    if failure_time is None:
        print('      Nenhuma falha de sensor detectada nos dados.')
        print('      Nenhuma previsão gerada.')
        return

    print(f'      Falha detectada em: {failure_time}')

    # 3. Treinar modelo com dados até o momento da falha
    print('\n[3/4] Treinando modelos...')
    df_treino = df[df['Data'] <= failure_time].copy()
    if len(df_treino) < MIN_TRAINING_SAMPLES:
        print('      Dados de treino insuficientes. Usando dataset completo.')
        df_treino = df.copy()

    model, scaler, use_scaler, model_name = train_best_model(df_treino)

    # 4. Inferência autorregressiva
    print(f'\n[4/4] Executando inferência autorregressiva ({PASSOS} passos × 15 min)...')
    tabela = run_autoregressive_forecast(
        df, model, scaler, use_scaler, failure_time, steps=PASSOS
    )

    # Exportar resultados
    tabela.to_excel(arquivo_saida, index=False)
    print(f'\n  Previsões exportadas: {arquivo_saida}')
    print('=' * 60)


if __name__ == '__main__':
    main()
