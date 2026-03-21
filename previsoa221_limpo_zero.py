"""
Previsão de Nível - UTR 221
============================
Modelo de Machine Learning para prever o nível do reservatório UTR-221
com base nos dados de bombas e nível UTR-220.

Dados: UTR221_2025_2026_15MIN.xls (intervalos de 15 minutos, 2025-2026)
Target: NIVEL-UTR-221 (medido em metros)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO = os.path.join(BASE_DIR, 'UTR221_2025_2026_15MIN.xls')

df = pd.read_excel(ARQUIVO, header=None, skiprows=2)
if df.shape[1] == 6:
    df.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                  'NIVEL-UTR-220', 'NIVEL-UTR-221', 'Obs']
    df = df.drop('Obs', axis=1)
elif df.shape[1] == 5:
    df.columns = ['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
                  'NIVEL-UTR-220', 'NIVEL-UTR-221']
else:
    raise ValueError(
        f'Formato inesperado no arquivo {os.path.basename(ARQUIVO)}: '
        f'{df.shape[1]} colunas encontradas.'
    )

colunas_numericas = [
    'CMB01-UTR-220-ESTADO',
    'CMB02-UTR-220-ESTADO',
    'NIVEL-UTR-220',
    'NIVEL-UTR-221',
]

for coluna in colunas_numericas:
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M', errors='coerce')

df = df.dropna(subset=['Data'] + colunas_numericas).reset_index(drop=True)
df = df.sort_values('Data').reset_index(drop=True)

# Calcular tendências (lags temporários) para obter a média antes da imputação
_lag1_221 = df['NIVEL-UTR-221'].shift(1)
_lag2_221 = df['NIVEL-UTR-221'].shift(2)
_lag1_220 = df['NIVEL-UTR-220'].shift(1)
_lag2_220 = df['NIVEL-UTR-220'].shift(2)

_tend_221 = _lag1_221 - _lag2_221
_tend_220 = _lag1_220 - _lag2_220

# Média das tendências calculada apenas onde ambos os lags são > 0
mask_221 = (_lag1_221 > 0) & (_lag2_221 > 0)
mask_220 = (_lag1_220 > 0) & (_lag2_220 > 0)
media_tend_221 = _tend_221[mask_221].mean()
media_tend_220 = _tend_220[mask_220].mean()

# Imputação de zeros usando a média da Tendencia_Nivel (features)
for col, media_tend in [('NIVEL-UTR-220', media_tend_220), ('NIVEL-UTR-221', media_tend_221)]:
    zeros_idx = df[df[col] == 0].index
    print(f'{col}: {len(zeros_idx)} zeros | média Tendência = {media_tend:.4f}')
    for idx in zeros_idx:
        if idx == 0:
            prox = df.loc[df.index > idx, col]
            prox_valido = prox[prox > 0]
            df.at[idx, col] = prox_valido.iloc[0] if not prox_valido.empty else df.loc[df[col] > 0, col].mean()
        else:
            df.at[idx, col] = df.at[idx - 1, col] + media_tend

# Exportar dados com zeros preenchidos
ARQUIVO_PREENCHIDO = os.path.join(BASE_DIR, 'UTR221_zeros_preenchidos.xlsx')
df[['Data', 'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
    'NIVEL-UTR-220', 'NIVEL-UTR-221']].to_excel(ARQUIVO_PREENCHIDO, index=False)
print(f'Dados com zeros preenchidos exportados: {ARQUIVO_PREENCHIDO}')

# =============================================================================
# 2. ENGENHARIA DE FEATURES (extraindo informações temporais)
# =============================================================================
df['Hora'] = df['Data'].dt.hour
df['Minuto'] = df['Data'].dt.minute
# One-Hot Encoding do dia da semana (7 colunas binárias)
_dias = pd.get_dummies(df['Data'].dt.dayofweek, prefix='DiaSemana').astype(int)
# Garante que as 7 colunas existam mesmo que algum dia não apareça nos dados
DIAS_SEMANA_COLS = [f'DiaSemana_{i}' for i in range(7)]
for c in DIAS_SEMANA_COLS:
    if c not in _dias.columns:
        _dias[c] = 0
_dias = _dias[DIAS_SEMANA_COLS]
df = pd.concat([df, _dias], axis=1)
df['Dia'] = df['Data'].dt.day
df['Periodo'] = df['Hora'].apply(
    lambda h: 0 if 0 <= h < 6 else (1 if 6 <= h < 12 else (2 if 12 <= h < 18 else 3))
)

# Features de lag (atrasos temporais)
df['NIVEL221_lag1'] = df['NIVEL-UTR-221'].shift(1)
df['NIVEL221_lag2'] = df['NIVEL-UTR-221'].shift(2)
df['NIVEL221_lag4'] = df['NIVEL-UTR-221'].shift(4)
df['NIVEL220_lag1'] = df['NIVEL-UTR-220'].shift(1)

# --- CORREÇÃO PASSO B: Tendência baseada apenas em dados passados ---
df['Tendencia_Nivel221'] = df['NIVEL221_lag1'] - df['NIVEL221_lag2']
df['Tendencia_Nivel220'] = df['NIVEL-UTR-220'].shift(1) - df['NIVEL-UTR-220'].shift(2)

df = df.dropna().reset_index(drop=True)

# =============================================================================
# 3. DEFINIÇÃO DE FEATURES E TARGET
# =============================================================================
TARGET = 'NIVEL-UTR-221'

# Lista atualizada para remover as variáveis com leakage
DIAS_SEMANA_COLS = [f'DiaSemana_{i}' for i in range(7)]
FEATURES = [
    'CMB01-UTR-220-ESTADO', 'CMB02-UTR-220-ESTADO',
    'NIVEL-UTR-220',
    'Hora', 'Minuto', 'Dia', 'Periodo',
] + DIAS_SEMANA_COLS + [
    'NIVEL221_lag1', 'NIVEL221_lag2', 'NIVEL221_lag4',
    'NIVEL220_lag1',
    'Tendencia_Nivel221',
    'Tendencia_Nivel220'
]

X = df[FEATURES]
y = df[TARGET]

# =============================================================================
# 4. DIVISÃO DOS DADOS (temporal - sem embaralhar)
# =============================================================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. TREINAMENTO DE MÚLTIPLOS MODELOS
# =============================================================================
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

resultados = {}

for nome, modelo in modelos.items():
    if nome in ['Linear Regression', 'Ridge Regression']:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    resultados[nome] = {'MAE': mae, 'R²': r2, 'Predições': y_pred}

# =============================================================================
# 6. COEFICIENTES DOS MODELOS
# =============================================================================
print('\n' + '=' * 60)
print('COEFICIENTES DOS MODELOS')
print('=' * 60)

for nome, modelo in modelos.items():
    print(f'\n--- {nome} ---')
    if hasattr(modelo, 'coef_'):
        print(f'  Intercepto: {modelo.intercept_:.6f}')
        for feat, coef in zip(FEATURES, modelo.coef_):
            print(f'  {feat:>25s}: {coef:+.6f}')
    elif hasattr(modelo, 'feature_importances_'):
        for feat, imp in zip(FEATURES, modelo.feature_importances_):
            barra = '█' * int(imp * 40)
            print(f'  {feat:>25s}: {imp:.6f} {barra}')

# =============================================================================
# 9. SIMULAÇÃO: Previsão do próximo nível (atualizado com novas colunas)
# =============================================================================
ultimo = df.iloc[-1]
proxima_data = ultimo['Data'] + pd.Timedelta(minutes=15)

novo_registro = {
    'CMB01-UTR-220-ESTADO': int(ultimo['CMB01-UTR-220-ESTADO']),
    'CMB02-UTR-220-ESTADO': int(ultimo['CMB02-UTR-220-ESTADO']),
    'NIVEL-UTR-220': ultimo['NIVEL-UTR-220'],
    'Hora': proxima_data.hour,
    'Minuto': proxima_data.minute,
    'Dia': proxima_data.day,
    'Periodo': 0 if 0 <= proxima_data.hour < 6 else (
        1 if 6 <= proxima_data.hour < 12 else (
            2 if 12 <= proxima_data.hour < 18 else 3)),
    **{c: int(proxima_data.dayofweek == i) for i, c in enumerate(DIAS_SEMANA_COLS)},
    'NIVEL221_lag1': ultimo['NIVEL-UTR-221'],
    'NIVEL221_lag2': df.iloc[-2]['NIVEL-UTR-221'],
    'NIVEL221_lag4': df.iloc[-4]['NIVEL-UTR-221'],
    'NIVEL220_lag1': ultimo['NIVEL-UTR-220'],
    'Tendencia_Nivel221': ultimo['Tendencia_Nivel221'],
    'Tendencia_Nivel220': ultimo['Tendencia_Nivel220'],
}

novo_df = pd.DataFrame([novo_registro])
novo_df = novo_df[FEATURES]

print('=' * 60)
print('RESULTADOS DOS MODELOS')
print('=' * 60)

for nome, res in resultados.items():
    print(f'\n--- {nome} ---')
    print(f'  MAE: {res["MAE"]:.4f} m')
    print(f'  R²:  {res["R²"]:.4f}')

melhor_nome = max(resultados, key=lambda k: resultados[k]['R²'])
print(f'\nMelhor modelo: {melhor_nome} (R²={resultados[melhor_nome]["R²"]:.4f})')

# =============================================================================
# 8. EXPORTAÇÃO: REAL X PREVISTO (ÚLTIMAS 12H)
# =============================================================================
datas_teste = df['Data'].iloc[split_idx:].reset_index(drop=True)
real_teste = y_test.reset_index(drop=True)
previsto_teste = pd.Series(resultados[melhor_nome]['Predições'])

inicio_janela = datas_teste.max() - pd.Timedelta(hours=12)
filtro_12h = datas_teste >= inicio_janela

planilha_12h = pd.DataFrame({
    'Data': datas_teste[filtro_12h],
    'Real (m)': real_teste[filtro_12h].values,
    'Previsto (m)': previsto_teste[filtro_12h].values,
})
planilha_12h['Erro (m)'] = planilha_12h['Real (m)'] - planilha_12h['Previsto (m)']
planilha_12h['Erro Abs (m)'] = planilha_12h['Erro (m)'].abs()

ARQUIVO_SAIDA = os.path.join(BASE_DIR, 'real_vs_previsto_ultimas_12h.xlsx')
planilha_12h.to_excel(ARQUIVO_SAIDA, index=False)
print(f'\nPlanilha criada: {ARQUIVO_SAIDA}')
print(f'Registros exportados (últimas 12h): {len(planilha_12h)}')

print(f'\nPrevisões para {proxima_data}:')
for nome, modelo in modelos.items():
    if nome in ['Linear Regression', 'Ridge Regression']:
        pred = modelo.predict(scaler.transform(novo_df))[0]
    else:
        pred = modelo.predict(novo_df)[0]
    marcador = ' <- melhor' if nome == melhor_nome else ''
    print(f'  {nome:>20s}: {pred:.2f} m{marcador}')

# =============================================================================
# 10. INFERÊNCIA AUTORREGRESSIVA: próxima 1h a partir de 2026-02-14 10:45
# =============================================================================
MOMENTO_PERDA = pd.Timestamp('2026-02-14 10:45:00')
PASSOS = 4  # 4 x 15 min = 1 hora

hist = df[df['Data'] <= MOMENTO_PERDA].copy()
if len(hist) < 5:
    print(f'\nDados insuficientes antes de {MOMENTO_PERDA}.')
else:
    modelo_inferencia = modelos[melhor_nome]
    usar_scaler = melhor_nome in ['Linear Regression', 'Ridge Regression']

    niveis_conhecidos = list(hist['NIVEL-UTR-221'].iloc[-4:].values)

    print(f'\n{"=" * 60}')
    print(f'INFERÊNCIA AUTORREGRESSIVA ({melhor_nome})')
    print(f'Perda de dados em: {MOMENTO_PERDA}')
    print(f'Último nível real conhecido: {niveis_conhecidos[-1]:.2f} m')
    print(f'{"=" * 60}')

    previsoes_futuras = []
    prev_nivel220 = hist['NIVEL-UTR-220'].iloc[-1]

    for passo in range(1, PASSOS + 1):
        dt = MOMENTO_PERDA + pd.Timedelta(minutes=15 * passo)

        # CMB e NIVEL-220: usa dados reais se existirem
        linha_real = df[df['Data'] == dt]
        if not linha_real.empty:
            cmb01 = int(linha_real['CMB01-UTR-220-ESTADO'].values[0])
            cmb02 = int(linha_real['CMB02-UTR-220-ESTADO'].values[0])
            nivel220 = linha_real['NIVEL-UTR-220'].values[0]
        else:
            cmb01 = int(hist['CMB01-UTR-220-ESTADO'].iloc[-1])
            cmb02 = int(hist['CMB02-UTR-220-ESTADO'].iloc[-1])
            nivel220 = prev_nivel220

        # NIVEL-220 anterior para tendência
        nivel220_ant1 = prev_nivel220
        lr_ant2 = df[df['Data'] == (dt - pd.Timedelta(minutes=30))]
        nivel220_ant2 = lr_ant2['NIVEL-UTR-220'].values[0] if not lr_ant2.empty else nivel220_ant1

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
            'Periodo': 0 if 0 <= dt.hour < 6 else (
                1 if 6 <= dt.hour < 12 else (
                    2 if 12 <= dt.hour < 18 else 3)),
            'NIVEL221_lag1': lag1,
            'NIVEL221_lag2': lag2,
            'NIVEL221_lag4': lag4,
            'NIVEL220_lag1': nivel220_ant1,
            'Tendencia_Nivel221': lag1 - lag2,
            'Tendencia_Nivel220': nivel220_ant1 - nivel220_ant2,
        }

        entrada = pd.DataFrame([registro])[FEATURES]
        if usar_scaler:
            pred = modelo_inferencia.predict(scaler.transform(entrada))[0]
        else:
            pred = modelo_inferencia.predict(entrada)[0]

        previsoes_futuras.append(pred)
        prev_nivel220 = nivel220

        # Comparação com real (se disponível)
        real_val = linha_real['NIVEL-UTR-221'].values[0] if not linha_real.empty else None
        real_str = f'{real_val:.2f}' if real_val is not None else '—'
        print(f'  {dt}  |  Previsto: {pred:.2f} m  |  Real: {real_str} m')

    # Exportar para Excel
    tabela_inf = pd.DataFrame({
        'Data': [MOMENTO_PERDA + pd.Timedelta(minutes=15 * i) for i in range(1, PASSOS + 1)],
        'Previsto (m)': previsoes_futuras,
    })
    reais = []
    for dt in tabela_inf['Data']:
        lr = df[df['Data'] == dt]
        reais.append(lr['NIVEL-UTR-221'].values[0] if not lr.empty else np.nan)
    tabela_inf['Real (m)'] = reais
    tabela_inf['Erro (m)'] = tabela_inf['Real (m)'] - tabela_inf['Previsto (m)']

    ARQUIVO_INF = os.path.join(BASE_DIR, 'inferencia_1h_nivel221.xlsx')
    tabela_inf.to_excel(ARQUIVO_INF, index=False)
    print(f'\nPlanilha salva: {ARQUIVO_INF}')

df = df[(df['NIVEL-UTR-220'] > 0) & (df['NIVEL-UTR-221'] > 0)].reset_index(drop=True)