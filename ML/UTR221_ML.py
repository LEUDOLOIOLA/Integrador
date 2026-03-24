import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random

# 1. CARREGAMENTO
df = pd.read_excel('UTR221_2025_2026_15MIN.xls', header=1)
df.columns = ['Data', 'BOMBA_1', 'BOMBA_2', 'NIVEL-UTR-220', 'NIVEL-UTR-221']
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df = df.set_index('Data').sort_index()

# 2. ENGENHARIA DE FEATURES
# Criando o alvo: o nível que acontecerá daqui a 15 min (próxima linha)
df['target'] = df['NIVEL-UTR-221'].shift(-1)

# Criando Lags (o que aconteceu 15 min atrás)
df['nivel_221_lag1'] = df['NIVEL-UTR-221'].shift(1)
df['nivel_220_lag1'] = df['NIVEL-UTR-220'].shift(1)

# Tendência baseada em dados passados
# Variação entre os dois últimos registros (delta 15 min)
df['tendencia_15min'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(2)
# Variação na última hora (4 períodos de 15 min)
df['tendencia_1h'] = df['NIVEL-UTR-221'].shift(1) - df['NIVEL-UTR-221'].shift(4)
# Média móvel das últimas 4 leituras (1 hora)
df['media_movel_1h'] = df['NIVEL-UTR-221'].shift(1).rolling(window=4).mean()

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
            'tendencia_15min', 'tendencia_1h', 'media_movel_1h']
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
# 12 horas = 48 períodos de 15 minutos
janela = 48 

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