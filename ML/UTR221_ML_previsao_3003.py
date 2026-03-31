import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================
# 1. CARREGAMENTO DE DADOS
# ==============================================================

# Carregar urt220 (preditores)
df220 = pd.read_excel('urt220.xlsx', usecols=['E3TIMESTAMP', 'CMB_220_S1A_EST', 'CMB_220_S1B_EST', 'PIT_220_S01_000'])
ts220 = (df220['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d/%m/%y %H:%M:%S.%f')
df220 = df220.set_index('E3TIMESTAMP').sort_index()

# Carregar urt221 (target)
df221 = pd.read_excel('urt221.xlsx', usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
ts221 = (df221['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df221['E3TIMESTAMP'] = pd.to_datetime(ts221, format='%d/%m/%y %H:%M:%S.%f')
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# Resample para 1 minuto
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

# Merge por timestamp
df = df220.join(df221, how='inner')

# Filtrar período: a partir de 28-FEB-26 17:30
df = df.loc['2026-02-28 17:30:38':]

# Renomear colunas
df = df.rename(columns={
    'CMB_220_S1A_EST': 'BOMBA_1',
    'CMB_220_S1B_EST': 'BOMBA_2',
    'PIT_220_S01_000': 'NIVEL-UTR-220',
    'PIT_221_S01_000': 'NIVEL-UTR-221'
})

# ==============================================================
# 2. ENGENHARIA DE FEATURES
# ==============================================================
df['target'] = df['NIVEL-UTR-221'].shift(-1)
df['nivel_221_lag1'] = df['NIVEL-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)
df['tendencia_1min'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(2)
df['tendencia_1h'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(60)
df['media_movel_1h'] = df['NIVEL-UTR-221'].shift(1).rolling(window=60).mean()
df['hora'] = df.index.hour
df = df.dropna()

# DF2: sem zeros
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

features = ['BOMBA_1', 'BOMBA_2', 'nivel_220_lag1', 'nivel_221_lag1', 'hora',
            'tendencia_1min', 'tendencia_1h', 'media_movel_1h']

X2 = df2[features]
y2 = df2['target']

print(f"DF  (original):      {len(df)} linhas")
print(f"DF2 (sem target=0):  {len(df2)} linhas")

# ==============================================================
# 3. TREINAMENTO — divisão temporal 80/20
# ==============================================================
split_point = int(len(df2) * 0.8)
X_train, X_test = X2.iloc[:split_point], X2.iloc[split_point:]
y_train, y_test = y2.iloc[:split_point], y2.iloc[split_point:]

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Métricas gerais do modelo
previsoes_test = modelo.predict(X_test)
mae = mean_absolute_error(y_test, previsoes_test)
rmse = np.sqrt(mean_squared_error(y_test, previsoes_test))
r2 = r2_score(y_test, previsoes_test)
mask_nz = y_test != 0
mape = np.mean(np.abs((y_test[mask_nz] - previsoes_test[mask_nz]) / y_test[mask_nz])) * 100
acuracia = 100 - mape

print("\n" + "=" * 50)
print("   MÉTRICAS DO MODELO (conjunto de teste)")
print("=" * 50)
print(f"MAE:       {mae:.4f} metros")
print(f"RMSE:      {rmse:.4f} metros")
print(f"R²:        {r2:.4f}")
print(f"MAPE:      {mape:.2f}%")
print(f"Acurácia:  {acuracia:.2f}%")
print("=" * 50)

# ==============================================================
# 4. PREVISÃO FUTURA: 30/03/26 23:45 → 31/03/26 00:45
# ==============================================================

prev_inicio = pd.Timestamp('2026-03-30 23:45:00')
prev_fim    = pd.Timestamp('2026-03-31 00:45:00')

ultimo_ts = df.index[-1]
print(f"\nÚltimo dado real:      {ultimo_ts}")
print(f"Início da previsão:    {prev_inicio}")
print(f"Fim da previsão:       {prev_fim}")

# Número total de passos desde o último dado até o fim da janela
total_passos = int((prev_fim - ultimo_ts).total_seconds() / 60)
print(f"Total de passos (min): {total_passos}")

# Histórico inicial: últimos 60 min reais
historico = list(df['NIVEL-UTR-221'].tail(60).values)

# Valores constantes dos últimos dados conhecidos
bomba1_const   = df['BOMBA_1'].iloc[-1]
bomba2_const   = df['BOMBA_2'].iloc[-1]
nivel220_const = df['NIVEL-UTR-220'].iloc[-1]

print(f"BOMBA_1 (último):  {bomba1_const}")
print(f"BOMBA_2 (último):  {bomba2_const}")
print(f"NIVEL-220 (último): {nivel220_const:.4f}")

registros_futuro = []

for passo in range(1, total_passos + 1):
    proximo_ts = ultimo_ts + pd.Timedelta(minutes=passo)

    nivel_lag1 = historico[-1]
    nivel_lag2 = historico[-2]
    nivel_lag60 = historico[-60] if len(historico) >= 60 else historico[0]

    feat_values = {
        'BOMBA_1': bomba1_const,
        'BOMBA_2': bomba2_const,
        'nivel_220_lag1': nivel220_const,
        'nivel_221_lag1': nivel_lag1,
        'hora': proximo_ts.hour,
        'tendencia_1min': nivel_lag1 - nivel_lag2,
        'tendencia_1h': nivel_lag1 - nivel_lag60,
        'media_movel_1h': np.mean(historico[-60:]),
    }

    X_futuro = pd.DataFrame([feat_values])[features]
    previsao = max(modelo.predict(X_futuro)[0], 0)

    registros_futuro.append({
        'Data': proximo_ts,
        'Previsto_Futuro': round(previsao, 4),
        'Passo': passo,
        'Hora': proximo_ts.strftime('%H:%M'),
    })

    historico.append(previsao)

df_futuro = pd.DataFrame(registros_futuro).set_index('Data')

# Recortar apenas a janela de interesse: 30/03 23:45 → 31/03 00:45
df_janela = df_futuro.loc[prev_inicio:prev_fim].copy()

print(f"\nPrevisão gerada: {len(df_futuro)} passos no total")
print(f"Janela de interesse ({prev_inicio.strftime('%d/%m %H:%M')}–{prev_fim.strftime('%d/%m %H:%M')}): {len(df_janela)} pontos")

# ==============================================================
# 5. EXPORTAR EXCEL
# ==============================================================
df_janela.to_excel('DF7_previsao_3003_2345_0045.xlsx')
print("\n✅ Arquivo gerado: DF7_previsao_3003_2345_0045.xlsx")

print(f"\n{'Timestamp':>22s} | {'Previsto (m)':>12s} | {'Hora':>6s}")
print("-" * 50)
for idx, row in df_janela.iterrows():
    print(f"{str(idx):>22s} | {row['Previsto_Futuro']:12.4f} | {row['Hora']:>6s}")

# ==============================================================
# 6. GRÁFICO
# ==============================================================
# Última 2h real como contexto
janela_real_inicio = ultimo_ts - pd.Timedelta(hours=2)
dados_reais = df.loc[janela_real_inicio:ultimo_ts].copy()

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(dados_reais.index, dados_reais['NIVEL-UTR-221'],
        label='Nível Real (UTR-221) — últimas 2h', color='blue', linewidth=2)
ax.plot(df_janela.index, df_janela['Previsto_Futuro'],
        label=f'Previsão (30/03 23:45 → 31/03 00:45)', color='green',
        linewidth=2.5, linestyle='--', marker='o', markersize=4)

ax.axvline(ultimo_ts, color='red', linestyle=':', linewidth=1.5, alpha=0.8,
           label=f'Último dado real ({ultimo_ts.strftime("%d/%m %H:%M")})')
ax.axvspan(prev_inicio, prev_fim, alpha=0.08, color='green', label='Janela prevista')

ax.set_title('Previsão Futura UTR-221\n30/03/2026 23:45 → 31/03/2026 00:45',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Nível (metros)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('previsao_3003_2345_0045.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: previsao_3003_2345_0045.png")
plt.show()

print("\n" + "=" * 60)
print(f"  Previsão 30/03/26 23:45 → 31/03/26 00:45 concluída")
print(f"  Nível mínimo previsto: {df_janela['Previsto_Futuro'].min():.4f} m")
print(f"  Nível máximo previsto: {df_janela['Previsto_Futuro'].max():.4f} m")
print(f"  Nível médio previsto:  {df_janela['Previsto_Futuro'].mean():.4f} m")
print("=" * 60)
