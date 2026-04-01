Projeto final
Professor Tarik

PREDIÇÃO DE NÍVEL DE RESERVATÓRIOS A JUSANTE PARA CONTROLE DE BOMBAS A MONTANTE.

1. Carregamento dos Dados - Filtra os dados a partir de 28/02/2026 17:30.

2. Engenharia de Features — Gera três DataFrames: DF (original), DF2 (sem linhas com target=0 e lag1=0) e DF3 (linhas com pressão UTR-221 = 0).

3. Definição de Features e Target — Define as 8 features do modelo.
   
5. Divisão Temporal — Separa o DF2 em 80% treino e 20% teste, respeitando a ordem cronológica (sem shuffle).

6. Treinamento — Treina um modelo RandomForestRegressor com 100 árvores.

7. Simulação (Previsão) — Gera previsões no conjunto de teste e exibe a importância de cada feature (coeficientes).

8. Métricas de Acurácia — Calcula e exibe MAE, RMSE, R², MAPE e Acurácia (100 − MAPE).

9. Previsão Futura de 7 Horas —  Realiza predição recursiva para os próximos 420 minutos (7h) a partir do último timestamp real. Simula a lógica de controle das bombas (liga/desliga por setpoints) e alimenta cada previsão como entrada do passo seguinte (feed-forward). Exporta o resultado para DF6_futuro_7h.xlsx e gera um gráfico com a última 1h real + 7h futuras.
