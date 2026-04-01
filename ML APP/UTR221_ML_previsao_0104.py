import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

# ============================================================
# PREVISÃO: 15 MINUTOS A PARTIR DE 02:02 EM 01/04/2026
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CARREGAMENTO DOS DADOS
df220 = pd.read_excel(os.path.join(BASE_DIR, 'urt220.xlsx'),
                      usecols=['E3TIMESTAMP', 'CMB_220_S2A_EST', 'CMB_220_S2B_EST', 'LIT_220_RA2_000'])
ts220 = (df220['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d/%m/%y %H:%M:%S.%f')
df220 = df220.set_index('E3TIMESTAMP').sort_index()

df221 = pd.read_excel(os.path.join(BASE_DIR, 'urt221.xlsx'),
                      usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
ts221 = (df221['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df221['E3TIMESTAMP'] = pd.to_datetime(ts221, format='%d/%m/%y %H:%M:%S.%f')
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# Resample para 1 minuto
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

df = df220.join(df221, how='inner')
df = df.loc['2026-02-28 17:30:38':]
df = df.rename(columns={
    'CMB_220_S2A_EST': 'BOMBA_1',
    'CMB_220_S2B_EST': 'BOMBA_2',
    'LIT_220_RA2_000': 'NIVEL-UTR-220',
    'PIT_221_S01_000': 'PRESSAO-UTR-221'
})

# 2. ENGENHARIA DE FEATURES
df['target'] = df['PRESSAO-UTR-221'].shift(-1)
df['nivel_221_lag1'] = df['PRESSAO-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)
df['tendencia_1min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(2)
df['tendencia_1h'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(60)
df['media_movel_1h'] = df['PRESSAO-UTR-221'].shift(1).rolling(window=60).mean()
df['hora'] = df.index.hour
df = df.dropna()

# Treinar sem zeros no target
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

features = ['BOMBA_1', 'BOMBA_2', 'nivel_220_lag1', 'nivel_221_lag1', 'hora',
            'tendencia_1min', 'tendencia_1h', 'media_movel_1h']

# 3. TREINAMENTO COM TODOS OS DADOS DISPONÍVEIS
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(df2[features], df2['target'])
print(f"Modelo treinado com {len(df2)} amostras")
print(f"Último dado real disponível: {df.index[-1]}")

# 4. CONFIGURAÇÃO DA PREVISÃO
INICIO_PREVISAO = pd.Timestamp('2026-04-01 02:02:00')
PASSOS = 15  # 15 minutos

# Verificar se há dados reais até o início da previsão
dados_ate_inicio = df[df.index <= INICIO_PREVISAO]
if len(dados_ate_inicio) == 0:
    print(f"⚠️  Sem dados até {INICIO_PREVISAO}. Usando último dado disponível como referência.")
    dados_ate_inicio = df

ultimo_ts = dados_ate_inicio.index[-1]
print(f"Referência para previsão: {ultimo_ts}")

# Histórico das últimas 60 leituras reais (para features recursivas)
historico = list(dados_ate_inicio['PRESSAO-UTR-221'].tail(60).values)

bomba1 = dados_ate_inicio['BOMBA_1'].iloc[-1]
bomba2 = dados_ate_inicio['BOMBA_2'].iloc[-1]
nivel220_const = dados_ate_inicio['NIVEL-UTR-220'].iloc[-1]

SETPOINT_ALTO = 1.68
SETPOINT_BAIXO = 1.55

# 5. PREVISÃO RECURSIVA 15 MINUTOS
print("\n" + "=" * 60)
print(f"   PREVISÃO: 15 MIN A PARTIR DE {INICIO_PREVISAO.strftime('%H:%M %d/%m/%Y')}")
print("=" * 60)

registros = []
ts_atual = INICIO_PREVISAO - pd.Timedelta(minutes=1)  # base para o shift de +1

for passo in range(1, PASSOS + 1):
    proximo_ts = INICIO_PREVISAO + pd.Timedelta(minutes=passo - 1)

    nivel_lag1 = historico[-1]
    nivel_lag2 = historico[-2] if len(historico) >= 2 else historico[-1]
    nivel_lag60 = historico[-60] if len(historico) >= 60 else historico[0]

    if nivel_lag1 >= SETPOINT_ALTO:
        bomba1, bomba2 = 0, 0
    elif nivel_lag1 <= SETPOINT_BAIXO:
        bomba1, bomba2 = 1, 1

    feat_values = {
        'BOMBA_1': bomba1,
        'BOMBA_2': bomba2,
        'nivel_220_lag1': nivel220_const,
        'nivel_221_lag1': nivel_lag1,
        'hora': proximo_ts.hour,
        'tendencia_1min': nivel_lag1 - nivel_lag2,
        'tendencia_1h': nivel_lag1 - nivel_lag60,
        'media_movel_1h': np.mean(historico[-60:]),
    }

    X_futuro = pd.DataFrame([feat_values])[features]
    previsao = max(modelo.predict(X_futuro)[0], 0)

    registros.append({
        'Data': proximo_ts,
        'Previsto': round(previsao, 4),
        'BOMBA_1': int(bomba1),
        'BOMBA_2': int(bomba2),
        'Hora': proximo_ts.strftime('%H:%M'),
    })

    historico.append(previsao)
    print(f"  {proximo_ts.strftime('%H:%M')}  →  {previsao:.4f} BAR  (B1={int(bomba1)} B2={int(bomba2)})")

df_prev = pd.DataFrame(registros).set_index('Data')
print("=" * 60)

# 6. EXPORTAR EXCEL
out_xlsx = os.path.join(BASE_DIR, 'DF7_previsao_0104_0202_0217.xlsx')
df_prev.to_excel(out_xlsx)
print(f"\n✅ Excel salvo: {out_xlsx}")

# 7. GRÁFICO: última 1h real + 15 min previstos
janela_inicio = ultimo_ts - pd.Timedelta(hours=1)
dados_grafico = df.loc[janela_inicio:ultimo_ts].copy()

plt.figure(figsize=(14, 6))
plt.plot(dados_grafico.index, dados_grafico['PRESSAO-UTR-221'],
         label='Dados Reais (última 1h)', color='royalblue', linewidth=2)
plt.plot(df_prev.index, df_prev['Previsto'],
         label='Previsão 15 min', color='darkorange', linewidth=2, linestyle='--', marker='o', markersize=5)
plt.axvline(INICIO_PREVISAO, color='red', linestyle=':', alpha=0.8, label=f'Início previsão {INICIO_PREVISAO.strftime("%H:%M")}')
plt.title(f'Previsão 15 Minutos — UTR-221\nA partir de {INICIO_PREVISAO.strftime("%H:%M %d/%m/%Y")}', fontsize=13)
plt.xlabel('Timestamp')
plt.ylabel('Pressão (BAR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

out_png = os.path.join(BASE_DIR, 'previsao_0104_0202_0217.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico salvo: {out_png}")
plt.show()
print("=" * 60)
