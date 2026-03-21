# Previsão de Nível - UTR 221

Projeto de Machine Learning para prever o nível do reservatório UTR-221 com base em dados de bombas e do nível do reservatório UTR-220, utilizando registros em intervalos de 15 minutos.

## Arquivos do Projeto

### Notebook / Script Principal

| Arquivo | Descrição |
| --- | --- |
| `previsao_nivel_221.ipynb` | **Notebook Jupyter** principal (recomendado). Contém todas as etapas divididas em células com cabeçalhos Markdown: carregamento e limpeza dos dados, imputação de zeros, engenharia de features, treinamento e comparação de modelos, exportação dos resultados e inferência autorregressiva para 1 hora após perda de sinal. |
| `previsoa221_limpo_zero.py` | Script Python equivalente (versão legada). |

### Dados de Entrada

| Arquivo | Descrição |
| --- | --- |
| `UTR221_2025_2026_15MIN.xls` | Dados brutos originais com registros a cada 15 minutos (2025–2026). Contém as colunas: Data, CMB01-UTR-220-ESTADO, CMB02-UTR-220-ESTADO, NIVEL-UTR-220 e NIVEL-UTR-221. |

### Dados Gerados pelo Script

| Arquivo | Descrição |
| --- | --- |
| `UTR221_dados_limpos.xlsx` | Dados após limpeza inicial (remoção de linhas inválidas e conversão de tipos). |
| `UTR221_zeros_preenchidos.xlsx` | Dados com os valores zero de NIVEL-UTR-220 e NIVEL-UTR-221 substituídos por estimativas baseadas na tendência média do nível. |
| `real_vs_previsto_ultimas_12h.xlsx` | Comparação entre valores reais e valores previstos pelo melhor modelo nas últimas 12 horas do conjunto de teste. Inclui colunas de erro absoluto. |
| `inferencia_1h_nivel221.xlsx` | Previsões autorregressivas do nível UTR-221 para os 4 intervalos de 15 minutos seguintes ao momento de perda de dados (2026-02-14 10:45), com comparação com valores reais quando disponíveis. |

## Modelos Utilizados

- **Linear Regression** – Regressão linear simples
- **Ridge Regression** (α = 1.0) – Regressão com regularização L2
- **Random Forest** (100 árvores) – Ensemble de árvores de decisão
- **Gradient Boosting** (100 estimadores) – Boosting com árvores de decisão

## Como Executar

### Notebook (recomendado)

```bash
pip install pandas numpy scikit-learn openpyxl xlrd matplotlib notebook
jupyter notebook previsao_nivel_221.ipynb
```

Execute as células em ordem. O notebook exibe resultados intermediários em cada célula e gera os arquivos `.xlsx` de saída.

### Script Python (legado)

```bash
pip install pandas numpy scikit-learn openpyxl xlrd matplotlib
python previsoa221_limpo_zero.py
```

## Features Utilizadas

- Estado das bombas CMB01 e CMB02 (UTR-220)
- Nível UTR-220 (atual e com 1 passo de lag)
- Nível UTR-221 (lags de 1, 2 e 4 passos)
- Tendência do nível UTR-220 e UTR-221
- Hora, Minuto, Dia da semana (One-Hot), Período do dia
