import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# UTR221 – PREVISÃO DURANTE FALHA DE COMUNICAÇÃO
# Entrada via terminal: lags de nível/status UTR220 +
#                       última pressão UTR221 + horário da falha
# Saída: gráfico de 6 horas de previsão
# ================================================================

# ----------------------------------------------------------------
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS (treino)
# ----------------------------------------------------------------
print("=" * 60)
print("  CARREGANDO DADOS HISTÓRICOS PARA TREINO DO MODELO...")
print("=" * 60)

# UTR220 – preditores
df220 = pd.read_excel('urt220.xlsx',
                      usecols=['E3TIMESTAMP',
                               'CMB_220_S2A_EST',   # Status B1
                               'CMB_220_S2B_EST',   # Status B2
                               'LIT_220_RA2_000'])  # Nível
ts220 = (df220['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d/%m/%y %H:%M:%S.%f')
df220 = df220.set_index('E3TIMESTAMP').sort_index()

# UTR221 – variável alvo
df221 = pd.read_excel('urt221.xlsx',
                      usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
ts221 = (df221['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df221['E3TIMESTAMP'] = pd.to_datetime(ts221, format='%d/%m/%y %H:%M:%S.%f')
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# Resample para 1 minuto
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

# Merge e filtro de período
df = df220.join(df221, how='inner')
df = df.loc['2026-02-28 17:30:38':]
df = df.rename(columns={
    'CMB_220_S2A_EST': 'BOMBA_1',
    'CMB_220_S2B_EST': 'BOMBA_2',
    'LIT_220_RA2_000': 'NIVEL_220',
    'PIT_221_S01_000': 'PRESSAO_221'
})

# ----------------------------------------------------------------
# 2. ENGENHARIA DE FEATURES
# ----------------------------------------------------------------
df['target']            = df['PRESSAO_221'].shift(-1)
df['pressao_221_lag1']  = df['PRESSAO_221'].shift(1)
df['nivel_220_lag1']    = df['NIVEL_220'].shift(1)
df['bomba1_lag1']       = df['BOMBA_1'].shift(1)
df['bomba2_lag1']       = df['BOMBA_2'].shift(1)
df['tendencia_1min']    = df['PRESSAO_221'].shift(1) - df['PRESSAO_221'].shift(2)
df['tendencia_1h']      = df['PRESSAO_221'].shift(1) - df['PRESSAO_221'].shift(60)
df['media_movel_1h']    = df['PRESSAO_221'].shift(1).rolling(window=60).mean()
df['hora']              = df.index.hour
df = df.dropna()

# Remover períodos com pressão = 0
df2 = df[(df['target'] != 0) & (df['pressao_221_lag1'] != 0)].copy()

FEATURES = [
    'bomba1_lag1', 'bomba2_lag1',
    'nivel_220_lag1',
    'pressao_221_lag1',
    'hora',
    'tendencia_1min',
    'tendencia_1h',
    'media_movel_1h',
]

# ----------------------------------------------------------------
# 3. TREINO DO MODELO
# ----------------------------------------------------------------
split = int(len(df2) * 0.8)
X_train = df2[FEATURES].iloc[:split]
y_train = df2['target'].iloc[:split]
X_test  = df2[FEATURES].iloc[split:]
y_test  = df2['target'].iloc[split:]

print("Treinando RandomForestRegressor...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

previsoes_teste = modelo.predict(X_test)
mae  = mean_absolute_error(y_test, previsoes_teste)
rmse = np.sqrt(mean_squared_error(y_test, previsoes_teste))
r2   = r2_score(y_test, previsoes_teste)
mape = np.mean(np.abs((y_test[y_test != 0] - previsoes_teste[y_test != 0])
                      / y_test[y_test != 0])) * 100
print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Acurácia={100-mape:.2f}%")

# ----------------------------------------------------------------
# 4. ENTRADA DE DADOS VIA TERMINAL
# ----------------------------------------------------------------
print()
print("=" * 60)
print("  ENTRADA DE DADOS – MOMENTO DA FALHA DE COMUNICAÇÃO")
print("=" * 60)
print("  (Informe os valores no instante em que a comunicação")
print("   com a UTR221 foi perdida.)")
print()

def pedir_float(mensagem, default=None):
    while True:
        try:
            entrada = input(mensagem).strip()
            if entrada == '' and default is not None:
                return default
            return float(entrada)
        except ValueError:
            print("  ⚠  Valor inválido. Digite um número (ex: 1.65).")

def pedir_int_binario(mensagem):
    while True:
        try:
            v = int(input(mensagem).strip())
            if v in (0, 1):
                return v
            print("  ⚠  Digite 0 (desligado) ou 1 (ligado).")
        except ValueError:
            print("  ⚠  Valor inválido. Digite 0 ou 1.")

def pedir_datetime(mensagem):
    formatos = [
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%d/%m/%y %H:%M:%S',
        '%d/%m/%y %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
    ]
    while True:
        entrada = input(mensagem).strip()
        for fmt in formatos:
            try:
                return pd.to_datetime(entrada, format=fmt)
            except ValueError:
                continue
        print("  ⚠  Formato não reconhecido.")
        print("     Use: DD/MM/AAAA HH:MM  ou  AAAA-MM-DD HH:MM")

# Horário da falha
horario_falha = pedir_datetime(
    "Horário da falha de comunicação (DD/MM/AAAA HH:MM): ")

# Status das bombas na UTR220
print()
print("  – Status das bombas na UTR220 no momento da falha –")
b1_lag1 = pedir_int_binario("  Status B1 (CMB_220_S2A_EST) [0=desligado / 1=ligado]: ")
b2_lag1 = pedir_int_binario("  Status B2 (CMB_220_S2B_EST) [0=desligado / 1=ligado]: ")

# Nível UTR220
print()
print("  – Nível (LIT_220_RA2_000) na UTR220 –")
nivel_220_lag1 = pedir_float("  Último nível UTR220 [m]: ")

# Pressão UTR221
print()
print("  – Pressão (PIT_221_S01_000) na UTR221 –")
print("  (Informe os valores antes da falha de comunicação.)")
p_lag1 = pedir_float("  Pressão mais recente       (lag 1, há  1 min) [bar]: ")
p_lag2 = pedir_float("  Pressão anterior            (lag 2, há  2 min) [bar]: ")
p_lag60 = pedir_float("  Pressão há 60 min           (lag 60, há 1 hora) [bar]: ")
p_media = pedir_float("  Média de pressão última hora (60 min)           [bar]: ")

hora_falha = horario_falha.hour

print()
print("=" * 60)
print(f"  Horário da falha  : {horario_falha.strftime('%d/%m/%Y %H:%M')}")
print(f"  Status B1 (lag1)  : {b1_lag1}")
print(f"  Status B2 (lag1)  : {b2_lag1}")
print(f"  Nível UTR220      : {nivel_220_lag1:.4f} m")
print(f"  Pressão UTR221 L1 : {p_lag1:.4f} bar")
print(f"  Pressão UTR221 L2 : {p_lag2:.4f} bar")
print(f"  Pressão UTR221 L60: {p_lag60:.4f} bar")
print(f"  Média pressão 1h  : {p_media:.4f} bar")
print("=" * 60)

# ----------------------------------------------------------------
# 5. PREVISÃO RECURSIVA – 6 HORAS (360 min)
# ----------------------------------------------------------------
PASSOS = 360  # 6 horas

# Histórico de pressão (inicializado com lag60 e expandido até lag1)
historico = [p_lag60] * 59 + [p_lag2, p_lag1]  # 61 valores, mais recente = [-1]

# Estado inicial das bombas
bomba1 = float(b1_lag1)
bomba2 = float(b2_lag1)

# Setpoints de controle
SETPOINT_ALTO  = 1.68
SETPOINT_BAIXO = 1.55

registros = []

print("\n  Gerando previsão recursiva de 6 horas...")

for passo in range(1, PASSOS + 1):
    ts_prev = horario_falha + pd.Timedelta(minutes=passo)

    # Valores de lag para este passo
    pressao_lag1  = historico[-1]
    pressao_lag2  = historico[-2]
    pressao_lag60 = historico[-60] if len(historico) >= 60 else historico[0]
    med_1h        = float(np.mean(historico[-60:])) if len(historico) >= 60 else float(np.mean(historico))
    tend_1min     = pressao_lag1 - pressao_lag2
    tend_1h       = pressao_lag1 - pressao_lag60

    # Lógica de controle das bombas
    if pressao_lag1 >= SETPOINT_ALTO:
        bomba1, bomba2 = 0.0, 0.0
    elif pressao_lag1 <= SETPOINT_BAIXO:
        bomba1, bomba2 = 1.0, 1.0

    feat = {
        'bomba1_lag1':     bomba1,
        'bomba2_lag1':     bomba2,
        'nivel_220_lag1':  nivel_220_lag1,
        'pressao_221_lag1': pressao_lag1,
        'hora':            ts_prev.hour,
        'tendencia_1min':  tend_1min,
        'tendencia_1h':    tend_1h,
        'media_movel_1h':  med_1h,
    }

    X_pred = pd.DataFrame([feat])[FEATURES]
    pressao_prev = max(modelo.predict(X_pred)[0], 0.0)

    registros.append({
        'Data':           ts_prev,
        'Pressao_Prev':   round(pressao_prev, 4),
        'BOMBA_1':        int(bomba1),
        'BOMBA_2':        int(bomba2),
        'Tendencia_1min': round(tend_1min, 4),
        'Hora':           ts_prev.strftime('%H:%M'),
    })

    historico.append(pressao_prev)

df_prev = pd.DataFrame(registros).set_index('Data')

print(f"  ✅ {len(df_prev)} passos previstos (6 horas).")
print()
print(df_prev[['Pressao_Prev', 'BOMBA_1', 'BOMBA_2', 'Hora']].head(10).to_string())
print("  ...")
print(df_prev[['Pressao_Prev', 'BOMBA_1', 'BOMBA_2', 'Hora']].tail(10).to_string())

# ----------------------------------------------------------------
# 6. EXPORTAR RESULTADOS
# ----------------------------------------------------------------
xlsx_nome = f"previsao_falha_{horario_falha.strftime('%d%m_%H%M')}.xlsx"
df_prev.to_excel(xlsx_nome)
print(f"\n  ✅ Excel salvo: {xlsx_nome}")

# ----------------------------------------------------------------
# 7. GRÁFICO – 6 HORAS DE DURAÇÃO DA FALHA
# ----------------------------------------------------------------
png_nome = f"previsao_falha_{horario_falha.strftime('%d%m_%H%M')}.png"

fig, ax = plt.subplots(figsize=(16, 7))

# Linha de previsão
ax.plot(df_prev.index, df_prev['Pressao_Prev'],
        color='steelblue', linewidth=2, label='Pressão prevista UTR221')

# Área sob a curva
ax.fill_between(df_prev.index, df_prev['Pressao_Prev'],
                alpha=0.15, color='steelblue')

# Marcas de setpoints
ax.axhline(SETPOINT_ALTO,  color='red',    linestyle='--', linewidth=1.2,
           alpha=0.7, label=f'Setpoint alto  ({SETPOINT_ALTO} bar)')
ax.axhline(SETPOINT_BAIXO, color='orange', linestyle='--', linewidth=1.2,
           alpha=0.7, label=f'Setpoint baixo ({SETPOINT_BAIXO} bar)')

# Linha vertical no início da falha
ax.axvline(horario_falha, color='crimson', linestyle=':', linewidth=1.5,
           label=f'Início falha: {horario_falha.strftime("%d/%m/%Y %H:%M")}')

# Estado das bombas como marcações coloridas no eixo x
bomba_on = df_prev[df_prev['BOMBA_1'] + df_prev['BOMBA_2'] > 0].index
bomba_off = df_prev[df_prev['BOMBA_1'] + df_prev['BOMBA_2'] == 0].index
y_bottom = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0
for ts in bomba_on:
    ax.axvline(ts, color='green', linewidth=0.3, alpha=0.15)
for ts in bomba_off:
    ax.axvline(ts, color='gray', linewidth=0.3, alpha=0.08)

# Formatação do eixo de tempo
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
plt.xticks(rotation=45)

# Rótulo de data no título
data_str = horario_falha.strftime('%d/%m/%Y')
ax.set_title(
    f'Previsão de Pressão UTR221 durante Falha de Comunicação\n'
    f'Data: {data_str}  |  Início: {horario_falha.strftime("%H:%M")}  |  Duração: 6 horas',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Horário', fontsize=11)
ax.set_ylabel('Pressão (bar)', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, which='major', alpha=0.4)
ax.grid(True, which='minor', alpha=0.15)

# Anotação de status das bombas
ax2 = ax.twinx()
ax2.step(df_prev.index,
         df_prev['BOMBA_1'].values.astype(float),
         where='post', color='green', linewidth=1.2,
         alpha=0.55, linestyle='-', label='B1 ligada')
ax2.step(df_prev.index,
         df_prev['BOMBA_2'].values.astype(float),
         where='post', color='limegreen', linewidth=1.2,
         alpha=0.55, linestyle='--', label='B2 ligada')
ax2.set_ylabel('Estado das Bombas (0/1)', fontsize=10)
ax2.set_ylim(-0.5, 3.5)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(png_nome, dpi=300, bbox_inches='tight')
print(f"  ✅ Gráfico salvo: {png_nome}")
plt.show()

print("\n" + "=" * 60)
print("  PREVISÃO CONCLUÍDA")
print("=" * 60)
