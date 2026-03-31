import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# 1. CARREGAMENTO
# ============================================================
df220 = pd.read_excel('urt220.xlsx', usecols=['E3TIMESTAMP', 'CMB_220_S1A_EST', 'CMB_220_S1B_EST', 'PIT_220_S01_000'])
ts220 = (df220['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d/%m/%y %H:%M:%S.%f')
df220 = df220.set_index('E3TIMESTAMP').sort_index()

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

# Filtrar período: a partir de 28-FEB-26 17:30 até o último dado disponível
df = df.loc['2026-02-28 17:30:38':]

# Renomear colunas
df = df.rename(columns={
    'CMB_220_S1A_EST': 'BOMBA_1',
    'CMB_220_S1B_EST': 'BOMBA_2',
    'PIT_220_S01_000': 'NIVEL-UTR-220',
    'PIT_221_S01_000': 'NIVEL-UTR-221'
})

# ============================================================
# 2. ENGENHARIA DE FEATURES
# ============================================================
df['target'] = df['NIVEL-UTR-221'].shift(-1)
df['nivel_221_lag1'] = df['NIVEL-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)
df['tendencia_1min'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(2)
df['tendencia_1h'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(60)
df['media_movel_1h'] = df['NIVEL-UTR-221'].shift(1).rolling(window=60).mean()
df['hora'] = df.index.hour
df = df.dropna()

# ============================================================
# 3. JANELA DE TESTE: 21/03/2026 14:00 → 20:00 (360 min)
# ============================================================
TESTE_INICIO = pd.Timestamp('2026-03-21 14:00:00')
TESTE_FIM    = pd.Timestamp('2026-03-21 20:00:00')

print("=" * 60)
print("   SIMULAÇÃO AVANÇADA — JANELA DE TESTE: 21/03/2026")
print(f"   {TESTE_INICIO.strftime('%H:%M')} → {TESTE_FIM.strftime('%H:%M')} (360 min / 6 horas)")
print("=" * 60)

# ============================================================
# 4. REMOÇÃO DOS DADOS DE TESTE DO TREINO
# ============================================================
# df_treino: todos os dados FORA da janela de teste
mask_teste = (df.index >= TESTE_INICIO) & (df.index <= TESTE_FIM)
df_treino = df[~mask_teste].copy()
df_teste  = df[mask_teste].copy()

print(f"\nTotal de registros (completo):  {len(df)}")
print(f"Registros de treino (sem teste): {len(df_treino)}")
print(f"Registros no período de teste:  {len(df_teste)}")

if len(df_teste) == 0:
    print("\nAVISO: Nenhum dado encontrado no período de teste 21/03/2026 14:00-20:00.")
    print("Verifique se os arquivos urt220.xlsx e urt221.xlsx cobrem essa data.")
    exit(1)

# Remover linhas onde target ou nivel_221_lag1 é zero (como feito no df2)
df2_treino = df_treino[(df_treino['target'] != 0) & (df_treino['nivel_221_lag1'] != 0)].copy()
print(f"Registros de treino (sem zeros): {len(df2_treino)}")

# ============================================================
# 5. FEATURES E TREINAMENTO
# ============================================================
features = ['BOMBA_1', 'BOMBA_2', 'nivel_220_lag1', 'nivel_221_lag1', 'hora',
            'tendencia_1min', 'tendencia_1h', 'media_movel_1h']

X_train = df2_treino[features]
y_train = df2_treino['target']

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
print("\nModelo treinado com sucesso (dados de treino excluem a janela de teste).")

# Importância das features
importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n" + "=" * 40)
print("   IMPORTÂNCIA DAS FEATURES")
print("=" * 40)
for feat, imp in importancias.items():
    barra = "█" * int(imp * 40)
    print(f"{feat:20s} {imp:.4f}  {barra}")
print("=" * 40)

# ============================================================
# 6. SIMULAÇÃO RECURSIVA NA JANELA DE TESTE
# Usa APENAS dados anteriores ao início do teste como histórico
# ============================================================
dados_antes_teste = df[df.index < TESTE_INICIO]

if len(dados_antes_teste) < 60:
    print("AVISO: Histórico insuficiente antes do período de teste.")
    exit(1)

historico_sim = list(dados_antes_teste['NIVEL-UTR-221'].iloc[-60:].values)
registros_sim = []

for idx, row in df_teste.iterrows():
    nivel_lag1  = historico_sim[-1]
    nivel_lag2  = historico_sim[-2]
    nivel_lag60 = historico_sim[-60] if len(historico_sim) >= 60 else historico_sim[0]

    feat_values = {
        'BOMBA_1':         row['BOMBA_1'],
        'BOMBA_2':         row['BOMBA_2'],
        'nivel_220_lag1':  row['nivel_220_lag1'],
        'nivel_221_lag1':  nivel_lag1,
        'hora':            idx.hour,
        'tendencia_1min':  nivel_lag1 - nivel_lag2,
        'tendencia_1h':    nivel_lag1 - nivel_lag60,
        'media_movel_1h':  np.mean(historico_sim[-60:]),
    }

    X_sim = pd.DataFrame([feat_values])[features]
    previsao = max(modelo.predict(X_sim)[0], 0)

    registros_sim.append({
        'Data':               idx,
        'Real':               row['NIVEL-UTR-221'],
        'Simulado':           round(previsao, 4),
        'BOMBA_1':            row['BOMBA_1'],
        'BOMBA_2':            row['BOMBA_2'],
        'NIVEL-UTR-220':      row['NIVEL-UTR-220'],
    })

    historico_sim.append(previsao)

df_sim = pd.DataFrame(registros_sim).set_index('Data')

# ============================================================
# 7. MÉTRICAS DA SIMULAÇÃO
# ============================================================
sim_real = df_sim['Real']
sim_prev = df_sim['Simulado']

sim_mae  = mean_absolute_error(sim_real, sim_prev)
sim_rmse = np.sqrt(mean_squared_error(sim_real, sim_prev))
sim_r2   = r2_score(sim_real, sim_prev)
mask_nz  = sim_real != 0
sim_mape = (np.mean(np.abs((sim_real[mask_nz] - sim_prev[mask_nz]) / sim_real[mask_nz])) * 100
            if mask_nz.sum() > 0 else 0)
sim_acuracia = 100 - sim_mape

print("\n" + "=" * 50)
print("   MÉTRICAS — SIMULAÇÃO 21/03/2026 14h-20h")
print("=" * 50)
print(f"MAE       (Erro Médio Absoluto):  {sim_mae:.4f} metros")
print(f"RMSE      (Raiz Erro Quadrático): {sim_rmse:.4f} metros")
print(f"R²        (Coef. Determinação):   {sim_r2:.4f}")
print(f"MAPE      (Erro % Médio Abs.):    {sim_mape:.2f}%")
print(f"Acurácia  (100 - MAPE):           {sim_acuracia:.2f}%")
print("=" * 50)

# ============================================================
# 8. GRÁFICO: REAL vs SIMULADO
# ============================================================
# Contexto: 1h antes e 1h depois da janela de teste
ctx_inicio = TESTE_INICIO - pd.Timedelta(hours=1)
ctx_fim    = TESTE_FIM    + pd.Timedelta(hours=1)
dados_ctx  = df.loc[ctx_inicio:ctx_fim].copy()

fig, ax = plt.subplots(figsize=(15, 7))

# Nível real completo (incluindo contexto fora da janela)
ax.plot(dados_ctx.index, dados_ctx['NIVEL-UTR-221'],
        label='Nível Real (UTR-221)', color='blue', linewidth=1.8)

# Simulação apenas dentro da janela de teste
ax.plot(df_sim.index, df_sim['Simulado'],
        label='Nível Simulado (modelo)', color='red',
        linestyle='--', linewidth=2)

# Destaque da janela de teste
ax.axvspan(TESTE_INICIO, TESTE_FIM, alpha=0.12, color='orange',
           label='Janela de teste (14h-20h)')
ax.axvline(TESTE_INICIO, color='orange', linestyle=':', linewidth=1.2)
ax.axvline(TESTE_FIM,    color='orange', linestyle=':', linewidth=1.2)

# Anotação de métricas no gráfico
metricas_txt = (f"MAE={sim_mae:.4f} m  |  RMSE={sim_rmse:.4f} m\n"
                f"R²={sim_r2:.4f}  |  Acurácia={sim_acuracia:.2f}%")
ax.text(0.01, 0.97, metricas_txt, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

ax.set_title('Simulação UTR-221 — 21/03/2026  14:00 → 20:00\n'
             '(Modelo treinado sem os dados desse período)',
             fontsize=13)
ax.set_xlabel('Hora')
ax.set_ylabel('Nível (metros)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

output_png = 'simulacao_21mar_14h_20h.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"\nGráfico salvo: {output_png}")
plt.show()

# ============================================================
# 9. EXPORTAR RESULTADOS
# ============================================================
output_xlsx = 'DF_simulacao_21mar_14h_20h.xlsx'
df_sim.to_excel(output_xlsx)
print(f"Planilha salva: {output_xlsx}")

# Tabela resumo (primeiros e últimos 10)
print(f"\n{'Hora':>20s} | {'Real':>10s} | {'Simulado':>10s} | {'Erro':>10s}")
print("-" * 60)
amostra = pd.concat([df_sim.head(10), df_sim.tail(10)])
for idx, row in amostra.iterrows():
    erro = abs(row['Real'] - row['Simulado'])
    print(f"{str(idx):>20s} | {row['Real']:10.4f} | {row['Simulado']:10.4f} | {erro:10.4f}")
print("=" * 60)
