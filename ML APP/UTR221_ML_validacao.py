import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. CARREGAMENTO
# Carregar urt220 (preditores)
df220 = pd.read_excel('urt220.xlsx', usecols=['E3TIMESTAMP', 'CMB_220_S2A_EST', 'CMB_220_S2B_EST', 'LIT_220_RA2_000'])
# Formato no Excel: "01/02/26 00:00:27,954000000" → trocar vírgula por ponto, truncar p/ 6 dígitos
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

# Filtrar período: a partir de 28-FEB-26 17:30 até o último dado disponível
df = df.loc['2026-02-28 17:30:38':]

# Renomear colunas para manter compatibilidade com o modelo
df = df.rename(columns={
    'CMB_220_S2A_EST': 'BOMBA_1',
    'CMB_220_S2B_EST': 'BOMBA_2',
    'LIT_220_RA2_000': 'NIVEL-UTR-220',
    'PIT_221_S01_000': 'PRESSAO-UTR-221'
})

# 2. ENGENHARIA DE FEATURES
# Criando o alvo: o nível que acontecerá daqui a 1 min (próxima linha)
df['target'] = df['PRESSAO-UTR-221'].shift(-1)

# Criando Lags (o que aconteceu 1 min atrás)

df['nivel_221_lag1'] = df['PRESSAO-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)

# Tendência baseada em dados passados
# Variação entre os dois últimos registros (delta 1 min)
df['tendencia_1min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(2)
# Variação na última hora (60 períodos de 1 min)
df['tendencia_1h'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(60)
# Média móvel das últimas 60 leituras (1 hora)
df['media_movel_1h'] = df['PRESSAO-UTR-221'].shift(1).rolling(window=60).mean()

# Extraindo hora
df['hora'] = df.index.hour

# Remover a última linha que ficou com target vazio (NaN) devido ao shift
df = df.dropna()

# DF2: cópia sem linhas onde target é zero e sem transições (lag1=0)
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

# DF3: linhas onde PRESSAO-UTR-221 é zero
df3 = df[df['PRESSAO-UTR-221'] == 0].copy()

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

# ============================================================
# 9. PREVISÃO FUTURA 7 HORAS (A PARTIR DO ÚLTIMO TIMESTAMP)
# Predição recursiva para os próximos 420 minutos (7h)
# ============================================================

print("\n" + "=" * 70)
print("   PREVISÃO FUTURA: PRÓXIMAS 7 HORAS (420 min)")
print("=" * 70)

# Último timestamp real
ultimo_ts = df.index[-1]
print(f"Último dado real: {ultimo_ts}")

# Configurações
passos_futuro = 420  # 7 horas = 420 minutos
historico_inicial = list(df['PRESSAO-UTR-221'].tail(60).values)  # Última 1h real

# Valores iniciais para predição futura (últimos conhecidos)
bomba1 = df['BOMBA_1'].iloc[-1]
bomba2 = df['BOMBA_2'].iloc[-1]
nivel220_const = df['NIVEL-UTR-220'].iloc[-1]

# Setpoints de controle das bombas (nível UTR-221)
SETPOINT_ALTO = 1.68   # Nível em que as bombas desligam
SETPOINT_BAIXO = 1.55  # Nível em que as bombas religam

registros_futuro = []

for passo in range(1, passos_futuro + 1):
    # Próximo timestamp
    proximo_ts = ultimo_ts + pd.Timedelta(minutes=passo)
    
    # Dados do passo anterior para features
    nivel_lag1 = historico_inicial[-1]
    nivel_lag2 = historico_inicial[-2]
    nivel_lag60 = historico_inicial[-60] if len(historico_inicial) >= 60 else historico_inicial[0]
    
    # Lógica de controle das bombas (faltou o rodízio)
    if nivel_lag1 >= SETPOINT_ALTO:
        bomba1, bomba2 = 0, 0  # Desliga bombas
    elif nivel_lag1 <= SETPOINT_BAIXO:
        bomba1, bomba2 = 1, 1  # Liga bombas
    
    feat_values = {
        'BOMBA_1': bomba1,
        'BOMBA_2': bomba2,
        'nivel_220_lag1': nivel220_const,
        'nivel_221_lag1': nivel_lag1,
        'hora': proximo_ts.hour,
        'tendencia_1min': nivel_lag1 - nivel_lag2,
        'tendencia_1h': nivel_lag1 - nivel_lag60,
        'media_movel_1h': np.mean(historico_inicial[-60:]),
    }
    
    X_futuro = pd.DataFrame([feat_values])[features]
    previsao = max(modelo.predict(X_futuro)[0], 0)
    
    registros_futuro.append({
        'Data': proximo_ts,
        'Previsto_Futuro': round(previsao, 2),
        'BOMBA_1': bomba1,
        'BOMBA_2': bomba2,
        'Passo': passo,
        'Hora': proximo_ts.strftime('%H:%M'),
    })
    
    # Atualizar histórico (feed-forward recursivo)
    historico_inicial.append(previsao)

# Criar DataFrame futuro
df_futuro = pd.DataFrame(registros_futuro).set_index('Data')
print(f"Previsão gerada: {len(df_futuro)} passos (7h futuras)")
print(df_futuro[['Previsto_Futuro', 'Hora']].head(10))
print(df_futuro[['Previsto_Futuro', 'Hora']].tail(10))

# Exportar
df_futuro.to_excel('DF6_futuro_7h.xlsx')
print("✅ Arquivo gerado: DF6_futuro_7h.xlsx")

# Gráfico: última 1h real + 2h futuro
janela_grafico_inicio = ultimo_ts - pd.Timedelta(hours=1)
dados_grafico = df.loc[janela_grafico_inicio:ultimo_ts].copy()

plt.figure(figsize=(15, 8))
plt.plot(dados_grafico.index, dados_grafico['PRESSAO-UTR-221'], 
         label='Última 1h Real', color='blue', linewidth=2)
plt.plot(df_futuro.index, df_futuro['Previsto_Futuro'], 
         label='Previsão 7h Futuras', color='green', linewidth=2, linestyle='--')

plt.axvline(ultimo_ts, color='red', linestyle=':', alpha=0.7, label='Limite dos dados reais')

plt.title(f'Previsão Futura 7 Horas\nA partir de {ultimo_ts}', fontsize=14)
plt.xlabel('Timestamp')
plt.ylabel('Pressão (BAR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('previsao_futura_7h.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: previsao_futura_7h.png")
plt.show()

print("=" * 70)
