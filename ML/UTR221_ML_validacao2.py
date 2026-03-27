import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random

# 1. CARREGAMENTO
# Carregar urt220 (preditores)
df220 = pd.read_excel('urt220.xlsx', usecols=['E3TIMESTAMP', 'CMB_220_S1A_EST', 'CMB_220_S1B_EST', 'PIT_220_S01_000'])
ts220 = df220['E3TIMESTAMP'].astype(str).str.replace(r'(\.\d{6})\d+', r'\1', regex=True)
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d-%b-%y %I.%M.%S.%f %p')
df220 = df220.set_index('E3TIMESTAMP').sort_index()

# Carregar urt221 (target)
df221 = pd.read_excel('urt221.xlsx', usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
ts221 = df221['E3TIMESTAMP'].astype(str).str.replace(r'(\.\d{6})\d+', r'\1', regex=True)
df221['E3TIMESTAMP'] = pd.to_datetime(ts221, format='%d-%b-%y %I.%M.%S.%f %p')
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# Resample para 1 minuto
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

# Merge por timestamp
df = df220.join(df221, how='inner')

# Filtrar período: 28-FEB-26 17:30 até 26-MAR-26 08:37
df = df.loc['2026-02-28 17:30:38':'2026-03-26 08:37:55']

# Renomear colunas para manter compatibilidade com o modelo
df = df.rename(columns={
    'CMB_220_S1A_EST': 'BOMBA_1',
    'CMB_220_S1B_EST': 'BOMBA_2',
    'PIT_220_S01_000': 'NIVEL-UTR-220',
    'PIT_221_S01_000': 'NIVEL-UTR-221'
})

# 2. ENGENHARIA DE FEATURES
# Criando o alvo: o nível que acontecerá daqui a 1 min (próxima linha)
df['target'] = df['NIVEL-UTR-221'].shift(-1)

# Criando Lags (o que aconteceu 1 min atrás)
df['nivel_221_lag1'] = df['NIVEL-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)

# Tendência baseada em dados passados
# Variação entre os dois últimos registros (delta 1 min)
df['tendencia_1min'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(2)
# Variação na última hora (60 períodos de 1 min)
df['tendencia_1h'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(60)
# Média móvel das últimas 60 leituras (1 hora)
df['media_movel_1h'] = df['NIVEL-UTR-221'].shift(1).rolling(window=60).mean()

# Extraindo hora
df['hora'] = df.index.hour

# Remover a última linha que ficou com target vazio (NaN) devido ao shift
df = df.dropna()

# DF2: cópia sem linhas onde target é zero e sem transições (lag1=0)
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

# DF3: linhas onde NIVEL-UTR-221 é zero
df3 = df[df['NIVEL-UTR-221'] == 0].copy()

# 3. DEFINIÇÃO DE FEATURES E TARGET
features = ['BOMBA_1', 'BOMBA_2', 'nivel_220_lag1', 'nivel_221_lag1', 'hora',
            'tendencia_1min', 'tendencia_1h', 'media_movel_1h']
X = df[features]
y = df['target']

X2 = df2[features]
y2 = df2['target']

print(f"DF  (original):      {len(df)} linhas")
print(f"DF2 (sem target=0):  {len(df2)} linhas  ({len(df) - len(df2)} removidas)")
print(f"DF3 (NIVEL-221=0):   {len(df3)} linhas")

# 4. DIVISÃO TEMPORAL (Sem shuffle!)
split_point = int(len(df2) * 0.8)
X_train, X_test = X2.iloc[:split_point], X2.iloc[split_point:]
y_train, y_test = y2.iloc[:split_point], y2.iloc[split_point:]

# 5. TREINAMENTO
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. SIMULAÇÃO (Previsão)
previsoes = modelo.predict(X_test)

# 6.1 IMPORTÂNCIA DAS FEATURES (Coeficientes)
importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n" + "=" * 40)
print("   COEFICIENTES (Importância)")
print("=" * 40)
for feat, imp in importancias.items():
    barra = "█" * int(imp * 40)
    print(f"{feat:20s} {imp:.4f}  {barra}")
print("=" * 40)

# 7. MÉTRICAS DE ACURÁCIA
mae = mean_absolute_error(y_test, previsoes)
rmse = np.sqrt(mean_squared_error(y_test, previsoes))
r2 = r2_score(y_test, previsoes)
mape = np.mean(np.abs((y_test[y_test != 0] - previsoes[y_test != 0]) / y_test[y_test != 0])) * 100
acuracia = 100 - mape

print("=" * 40)
print("   MÉTRICAS DO MODELO")
print("=" * 40)
print(f"MAE  (Erro Médio Absoluto):  {mae:.4f} metros")
print(f"RMSE (Raiz do Erro Quadr.):  {rmse:.4f} metros")
print(f"R²   (Coef. Determinação):   {r2:.4f}")
print(f"MAPE (Erro % Médio Abs.):    {mape:.2f}%")
print(f"Acurácia (100 - MAPE):       {acuracia:.2f}%")
print("=" * 40)

# 1. PREPARAÇÃO DOS DADOS PARA O GRÁFICO
# Criamos um DataFrame para comparar os valores reais com as previsões
resultados = pd.DataFrame({
    'Real': y_test.values,
    'Previsto': previsoes
}, index=y_test.index)

# 2. SELEÇÃO DA JANELA ALEATÓRIA
# 12 horas = 720 períodos de 1 minuto
janela = 720 

# Escolhemos um ponto de início aleatório dentro do conjunto de teste
total_pontos = len(resultados)
inicio = random.randint(0, total_pontos - janela)
fim = inicio + janela

# Recortamos os dados para essas 12 horas específicas
exemplo_12h = resultados.iloc[inicio:fim]

# 3. CRIAÇÃO DO GRÁFICO
plt.figure(figsize=(12, 6))
plt.plot(exemplo_12h.index, exemplo_12h['Real'], label='Nível Real (UTR-221)', color='blue', marker='o')
plt.plot(exemplo_12h.index, exemplo_12h['Previsto'], label='Previsão do Modelo', color='red', linestyle='--', marker='x')

# Formatação e Legendas
plt.title(f'Simulação: Real vs Previsão (Janela de 12h iniciando em {exemplo_12h.index[0]})')
plt.xlabel('Hora da Medição')
plt.ylabel('Nível (metros)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar ou mostrar o gráfico
plt.savefig('comparativo_12h.png')
print(f"Gráfico gerado para o período: {exemplo_12h.index[0]} até {exemplo_12h.index[-1]}")
plt.show()

# ============================================================
# 8. VALIDAÇÃO RECURSIVA (DF4)
# Nos horários do DF3 (NIVEL-UTR-221=0), simula passo a passo
# onde cada previsão alimenta o próximo passo
# ============================================================

df3_sorted = df3.sort_index()
tempos = df3_sorted.index
gaps = (tempos[1:] - tempos[:-1]) > pd.Timedelta(minutes=1)
blocos_inicio = [tempos[0]] + list(tempos[1:][gaps])
blocos_fim = list(tempos[:-1][gaps]) + [tempos[-1]]

print("\n" + "=" * 50)
print("   VALIDAÇÃO RECURSIVA (DF4)")
print("=" * 50)
print(f"Blocos de zeros encontrados: {len(blocos_inicio)}")

registros_df4 = []

for b_inicio, b_fim in zip(blocos_inicio, blocos_fim):
    bloco = df.loc[(df.index >= b_inicio) & (df.index <= b_fim)]
    if len(bloco) == 0:
        continue

    anteriores = df.loc[df.index < b_inicio, 'NIVEL-UTR-221']
    if len(anteriores) < 60:
        continue

    historico = list(anteriores.iloc[-60:].values)

    for idx, row in bloco.iterrows():
        nivel_lag1 = historico[-1]
        nivel_lag2 = historico[-2]
        nivel_lag60 = historico[-60] if len(historico) >= 60 else historico[0]

        feat_values = {
            'BOMBA_1': row['BOMBA_1'],
            'BOMBA_2': row['BOMBA_2'],
            'nivel_220_lag1': row['nivel_220_lag1'],
            'nivel_221_lag1': nivel_lag1,
            'hora': idx.hour,
            'tendencia_1min': nivel_lag1 - nivel_lag2,
            'tendencia_1h': nivel_lag1 - nivel_lag60,
            'media_movel_1h': np.mean(historico[-60:]),
        }

        X_pred = pd.DataFrame([feat_values])[features]
        previsao = max(modelo.predict(X_pred)[0], 0)

        registros_df4.append({
            'Data': idx,
            'Real': row['NIVEL-UTR-221'],
            'Target_Real': row['target'],
            'Previsto_Recursivo': round(previsao, 4),
            'BOMBA_1': row['BOMBA_1'],
            'BOMBA_2': row['BOMBA_2'],
        })

        historico.append(previsao)

# Criar DF4
df4 = pd.DataFrame(registros_df4).set_index('Data')
print(f"DF4 (validação recursiva): {len(df4)} linhas")

# Acrescentar coluna Previsto_Recursivo no DF original
df['Previsto_Recursivo'] = df4['Previsto_Recursivo']

# Exportar DF2, DF3 e DF4 para Excel
df.to_excel('DF1_original.xlsx')
df2.to_excel('DF2_sem_zeros.xlsx')
df3.to_excel('DF3_nivel_zero.xlsx')
df4.to_excel('DF4_validacao_recursiva.xlsx')
print("\nArquivos Excel gerados: DF2_sem_zeros.xlsx, DF3_nivel_zero.xlsx, DF4_validacao_recursiva.xlsx")

print("=" * 50)

# ============================================================
# 9. SIMULAÇÃO DE FALHA DE COMUNICAÇÃO UTR-221 (06/mar 06:42-07:42)
# Simula perda de sinal: o modelo prevê recursivamente
# o nível usando apenas dados da UTR-220 e bombas
# ============================================================

print("\n" + "=" * 60)
print("   SIMULAÇÃO: FALHA DE COMUNICAÇÃO UTR-221 (06/MAR 06:42-07:42)")
print("=" * 60)

falha_inicio = pd.Timestamp('2026-03-06 06:42:00')
falha_fim = pd.Timestamp('2026-03-06 07:42:00')

print(f"Período da simulação: {falha_inicio} até {falha_fim}")

# Janela de contexto para o gráfico: 1h antes e 1h depois
janela_inicio = falha_inicio - pd.Timedelta(hours=1)
janela_fim = falha_fim + pd.Timedelta(hours=1)

dados_janela = df.loc[janela_inicio:janela_fim].copy()
dados_falha = df.loc[falha_inicio:falha_fim].copy()

print(f"Dados na janela de contexto: {len(dados_janela)} registros")
print(f"Dados no período de falha: {len(dados_falha)} registros")

# Recuperar histórico antes da falha (últimos 60 min reais)
dados_antes = df[df.index < falha_inicio]

if len(dados_antes) < 60 or len(dados_falha) == 0:
    print("AVISO: Dados insuficientes para simulação nesse intervalo.")
else:
    historico_sim = list(dados_antes['NIVEL-UTR-221'].iloc[-60:].values)
    registros_sim = []

    for idx, row in dados_falha.iterrows():
        nivel_lag1 = historico_sim[-1]
        nivel_lag2 = historico_sim[-2]
        nivel_lag60 = historico_sim[-60]

        feat_values = {
            'BOMBA_1': row['BOMBA_1'],
            'BOMBA_2': row['BOMBA_2'],
            'nivel_220_lag1': row['nivel_220_lag1'],
            'nivel_221_lag1': nivel_lag1,
            'hora': idx.hour,
            'tendencia_1min': nivel_lag1 - nivel_lag2,
            'tendencia_1h': nivel_lag1 - nivel_lag60,
            'media_movel_1h': np.mean(historico_sim[-60:]),
        }

        X_sim = pd.DataFrame([feat_values])[features]
        previsao_sim = max(modelo.predict(X_sim)[0], 0)

        registros_sim.append({
            'Data': idx,
            'Real': row['NIVEL-UTR-221'],
            'Previsto_Recursivo': round(previsao_sim, 4),
            'BOMBA_1': row['BOMBA_1'],
            'BOMBA_2': row['BOMBA_2'],
            'NIVEL-UTR-220': row['NIVEL-UTR-220'],
        })

        historico_sim.append(previsao_sim)

    df_sim = pd.DataFrame(registros_sim).set_index('Data')

    # Métricas da simulação
    sim_real = df_sim['Real']
    sim_prev = df_sim['Previsto_Recursivo']
    sim_mae = mean_absolute_error(sim_real, sim_prev)
    sim_rmse = np.sqrt(mean_squared_error(sim_real, sim_prev))
    sim_r2 = r2_score(sim_real, sim_prev)
    mask_nz = sim_real != 0
    sim_mape = np.mean(np.abs((sim_real[mask_nz] - sim_prev[mask_nz]) / sim_real[mask_nz])) * 100 if mask_nz.sum() > 0 else 0
    sim_acuracia = 100 - sim_mape

    print("\n--- Métricas da Simulação de Falha ---")
    print(f"MAE:       {sim_mae:.4f} metros")
    print(f"RMSE:      {sim_rmse:.4f} metros")
    print(f"R²:        {sim_r2:.4f}")
    print(f"MAPE:      {sim_mape:.2f}%")
    print(f"Acurácia:  {sim_acuracia:.2f}%")

    # Gráfico da simulação
    plt.figure(figsize=(14, 7))
    plt.plot(dados_janela.index, dados_janela['NIVEL-UTR-221'],
             label='Nível Real (UTR-221)', color='blue', linewidth=1.5)
    plt.plot(df_sim.index, df_sim['Previsto_Recursivo'],
             label='Previsão Recursiva (falha simulada)', color='red',
             linestyle='--', linewidth=2)
    plt.axvspan(falha_inicio, falha_fim, alpha=0.15, color='red',
                label='Período sem comunicação (06:42-07:42)')

    plt.title('Simulação de Falha UTR-221 — 06/MAR/2026 (06:42 às 07:42)')
    plt.xlabel('Hora')
    plt.ylabel('Nível (metros)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('simulacao_falha_06mar_0642_0742.png')
    print("\nGráfico salvo: simulacao_falha_06mar_0642_0742.png")
    plt.show()

    # Exportar simulação
    df_sim.to_excel('DF5_simulacao_falha_06mar_0642_0742.xlsx')
    print("Arquivo Excel gerado: DF5_simulacao_falha_06mar_0642_0742.xlsx")

    # Tabela resumo
    print(f"\n{'Hora':>20s} | {'Real':>10s} | {'Previsto':>10s} | {'Erro':>10s}")
    print("-" * 60)
    for idx, row in df_sim.iterrows():
        erro = abs(row['Real'] - row['Previsto_Recursivo'])
        print(f"{str(idx):>20s} | {row['Real']:10.4f} | {row['Previsto_Recursivo']:10.4f} | {erro:10.4f}")
    print("=" * 60)