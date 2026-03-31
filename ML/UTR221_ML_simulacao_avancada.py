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

df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

df = df220.join(df221, how='inner')
df = df.loc['2026-02-28 17:30:38':]
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

df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

features = ['BOMBA_1', 'BOMBA_2', 'nivel_220_lag1', 'nivel_221_lag1', 'hora',
            'tendencia_1min', 'tendencia_1h', 'media_movel_1h']

print(f"DF  (original):      {len(df)} linhas")
print(f"DF2 (sem target=0):  {len(df2)} linhas  ({len(df) - len(df2)} removidas)")

# ============================================================
# 3. SIMULAÇÃO AVANÇADA — TESTE 01
# Janela de teste: 06/MAR/2026 das 17h às 21h (período vespertino)
# Lógica de controle simulada:
#   - Nível <= 1.50 m → Bomba LIGA (BOMBA_1=1, BOMBA_2=1)
#   - Nível >= 1.68 m → Bomba DESLIGA (BOMBA_1=0, BOMBA_2=0)
#   - Entre os limiares → mantém estado atual (histerese)
# ============================================================

print("\n" + "=" * 65)
print("   SIMULAÇÃO AVANÇADA — TESTE 01 (14h – 20h)")
print("=" * 65)

NIVEL_LIGA    = 1.50   # nível que aciona a bomba
NIVEL_DESLIGA = 1.68   # nível que desliga a bomba

# --- 3.1 Definir a janela de teste: 21/MAR/2026 14:00 – 20:00 ---
inicio_teste = pd.Timestamp('2026-03-21 14:00:00')
fim_teste    = pd.Timestamp('2026-03-21 20:00:00')

nivel = df['NIVEL-UTR-221']

print(f"Janela de teste (período 14h–20h):")
print(f"  Início: {inicio_teste}  → nível real = {nivel.loc[inicio_teste]:.2f}m")
print(f"  Fim:    {fim_teste}  → nível real = {nivel.loc[fim_teste]:.2f}m")
print(f"  Duração: {fim_teste - inicio_teste}  |  "
      f"Linhas no teste: {len(df.loc[inicio_teste:fim_teste])}")

# --- 3.2 Remover a janela de teste dos dados de treino ---
df2_treino = df2[~((df2.index >= inicio_teste) & (df2.index <= fim_teste))].copy()

print(f"\nDF2 original:        {len(df2)} linhas")
print(f"DF2 sem janela teste:{len(df2_treino)} linhas  "
      f"({len(df2) - len(df2_treino)} linhas removidas da janela de teste)")

# --- 3.3 Treinar o modelo SEM a janela de teste ---
split_point = int(len(df2_treino) * 0.8)
X_train = df2_treino[features].iloc[:split_point]
y_train = df2_treino['target'].iloc[:split_point]
X_test  = df2_treino[features].iloc[split_point:]
y_test  = df2_treino['target'].iloc[split_point:]

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

previsoes_teste = modelo.predict(X_test)
mae  = mean_absolute_error(y_test, previsoes_teste)
rmse = np.sqrt(mean_squared_error(y_test, previsoes_teste))
r2   = r2_score(y_test, previsoes_teste)
mask_nz = y_test != 0
mape = np.mean(np.abs((y_test[mask_nz] - previsoes_teste[mask_nz]) / y_test[mask_nz])) * 100
acuracia = 100 - mape

print("\n--- Métricas do Modelo (treinado sem a janela de teste) ---")
print(f"MAE:       {mae:.4f} metros")
print(f"RMSE:      {rmse:.4f} metros")
print(f"R²:        {r2:.4f}")
print(f"MAPE:      {mape:.2f}%")
print(f"Acurácia:  {acuracia:.2f}%")

# --- 3.4 Simular a janela de teste com lógica de controle ---
# O estado da bomba é determinado pela lógica: liga em 1.50 / desliga em 1.68
# As features BOMBA_1 e BOMBA_2 são SUBSTITUÍDAS pelo estado simulado
dados_teste = df.loc[inicio_teste:fim_teste].copy()
dados_antes  = df[df.index < inicio_teste]

if len(dados_antes) < 60:
    print("\nAVISO: Histórico insuficiente antes da janela de teste.")
else:
    historico_sim = list(dados_antes['NIVEL-UTR-221'].iloc[-60:].values)

    # Estado inicial da bomba: se o último nível já <= NIVEL_LIGA, liga de imediato
    nivel_anterior = historico_sim[-1]
    estado_bomba   = 1 if nivel_anterior <= NIVEL_LIGA else 0

    registros_sim = []

    for idx, row in dados_teste.iterrows():
        nivel_lag1 = historico_sim[-1]
        nivel_lag2 = historico_sim[-2]
        nivel_lag60 = historico_sim[-60] if len(historico_sim) >= 60 else historico_sim[0]

        # Atualizar estado da bomba conforme lógica de controle
        if nivel_lag1 <= NIVEL_LIGA:
            estado_bomba = 1   # Liga
        elif nivel_lag1 >= NIVEL_DESLIGA:
            estado_bomba = 0   # Desliga
        # Else: mantém estado atual (histerese)

        feat_values = {
            'BOMBA_1': float(estado_bomba),
            'BOMBA_2': float(estado_bomba),
            'nivel_220_lag1': row['nivel_220_lag1'],
            'nivel_221_lag1': nivel_lag1,
            'hora': idx.hour,
            'tendencia_1min': nivel_lag1 - nivel_lag2,
            'tendencia_1h': nivel_lag1 - nivel_lag60,
            'media_movel_1h': np.mean(historico_sim[-60:]),
        }

        X_sim = pd.DataFrame([feat_values])[features]
        previsao = max(modelo.predict(X_sim)[0], 0)

        registros_sim.append({
            'Data': idx,
            'Real': row['NIVEL-UTR-221'],
            'Previsto_Simulado': round(previsao, 4),
            'BOMBA_Simulada': float(estado_bomba),
            'BOMBA_1_Real': row['BOMBA_1'],
            'BOMBA_2_Real': row['BOMBA_2'],
            'NIVEL-UTR-220': row['NIVEL-UTR-220'],
        })

        historico_sim.append(previsao)

    df_sim = pd.DataFrame(registros_sim).set_index('Data')

    # --- 3.5 Métricas da simulação avançada ---
    sim_real = df_sim['Real']
    sim_prev = df_sim['Previsto_Simulado']
    mask_nz_sim = sim_real != 0
    sim_mae  = mean_absolute_error(sim_real[mask_nz_sim], sim_prev[mask_nz_sim])
    sim_rmse = np.sqrt(mean_squared_error(sim_real[mask_nz_sim], sim_prev[mask_nz_sim]))
    sim_r2   = r2_score(sim_real[mask_nz_sim], sim_prev[mask_nz_sim])
    sim_mape = np.mean(np.abs(
        (sim_real[mask_nz_sim] - sim_prev[mask_nz_sim]) / sim_real[mask_nz_sim]
    )) * 100
    sim_acuracia = 100 - sim_mape

    print("\n--- Métricas da Simulação Avançada (janela de teste) ---")
    print(f"MAE:       {sim_mae:.4f} metros")
    print(f"RMSE:      {sim_rmse:.4f} metros")
    print(f"R²:        {sim_r2:.4f}")
    print(f"MAPE:      {sim_mape:.2f}%")
    print(f"Acurácia:  {sim_acuracia:.2f}%")

    # Resumo das transições da bomba simulada
    bomba_diff = df_sim['BOMBA_Simulada'].diff()
    ligas    = df_sim.index[bomba_diff > 0.5]
    desligas = df_sim.index[bomba_diff < -0.5]
    print(f"\nTransições da Bomba Simulada durante a janela de teste:")
    print(f"  Liga:    {len(ligas)} vezes → {ligas.tolist()}")
    print(f"  Desliga: {len(desligas)} vezes → {desligas.tolist()}")

    # --- 3.6 Gráfico: Real vs Simulado com estado da bomba ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Contexto: 30 min antes e 30 min depois da janela de teste
    ctx_inicio = inicio_teste - pd.Timedelta(minutes=30)
    ctx_fim    = fim_teste    + pd.Timedelta(minutes=30)
    dados_ctx  = df.loc[ctx_inicio:ctx_fim]

    # Painel superior: nível
    ax1.plot(dados_ctx.index, dados_ctx['NIVEL-UTR-221'],
             label='Nível Real (UTR-221)', color='blue', linewidth=1.5)
    ax1.plot(df_sim.index, df_sim['Previsto_Simulado'],
             label='Previsão Simulada (Bomba controlada por limiar)',
             color='red', linestyle='--', linewidth=2)
    ax1.axhline(NIVEL_LIGA,    color='green',  linestyle=':', alpha=0.8,
                label=f'Limiar de LIGA ({NIVEL_LIGA}m)')
    ax1.axhline(NIVEL_DESLIGA, color='orange', linestyle=':', alpha=0.8,
                label=f'Limiar de DESLIGA ({NIVEL_DESLIGA}m)')
    ax1.axvspan(inicio_teste, fim_teste, alpha=0.08, color='gray',
                label='Janela de Teste (removida do treino)')
    ax1.set_ylabel('Nível (metros)')
    ax1.set_title(
        f'Simulação Avançada — Teste 01 (Período 14h–20h)\n'
        f'Janela: {inicio_teste.strftime("%d/%m/%Y %H:%M")} a '
        f'{fim_teste.strftime("%d/%m/%Y %H:%M")}\n'
        f'Lógica: Liga ≤ {NIVEL_LIGA}m  |  Desliga ≥ {NIVEL_DESLIGA}m',
        fontsize=13
    )
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Painel inferior: estado da bomba
    ax2.step(df_sim.index, df_sim['BOMBA_Simulada'],
             label='Bomba Simulada', color='red', linewidth=1.5, where='post')
    ax2.step(dados_ctx.index, dados_ctx['BOMBA_1'],
             label='BOMBA_1 Real', color='blue', linewidth=1, where='post',
             linestyle='--', alpha=0.6)
    ax2.step(dados_ctx.index, dados_ctx['BOMBA_2'],
             label='BOMBA_2 Real', color='cyan', linewidth=1, where='post',
             linestyle=':', alpha=0.6)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.set_ylabel('Estado Bomba')
    ax2.set_xlabel('Hora')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(inicio_teste, fim_teste, alpha=0.08, color='gray')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('simulacao_avancada_teste01.png', dpi=150, bbox_inches='tight')
    print("\nGráfico salvo: simulacao_avancada_teste01.png")
    plt.show()

    # --- 3.7 Exportar resultados ---
    df_sim.to_excel('DF7_simulacao_avancada_teste01.xlsx')
    print("Arquivo Excel gerado: DF7_simulacao_avancada_teste01.xlsx")

    print("\n" + "=" * 65)
    print(f"{'Hora':>20s} | {'Real':>8s} | {'Previsto':>10s} | "
          f"{'Bomba Sim':>10s} | {'Bomba Real B1':>12s} | {'Erro':>8s}")
    print("-" * 80)
    for idx, row in df_sim.iterrows():
        erro = abs(row['Real'] - row['Previsto_Simulado'])
        print(f"{str(idx):>20s} | {row['Real']:8.4f} | "
              f"{row['Previsto_Simulado']:10.4f} | "
              f"{int(row['BOMBA_Simulada']):>10d} | "
              f"{int(row['BOMBA_1_Real']):>12d} | "
              f"{erro:8.4f}")
    print("=" * 65)
