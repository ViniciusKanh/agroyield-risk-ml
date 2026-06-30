# Metodologia experimental

## Problema

Classificar o risco de quebra de produtividade da soja em municípios brasileiros a partir de variáveis climáticas e histórico produtivo.

## Recorte inicial

- Cultura: soja.
- Estado: Paraná.
- Período: 2006 a 2024.
- Unidade de análise: município-ano.

## Variável-alvo

A variável-alvo `risco_quebra` é definida com base no rendimento médio da produção.

Para cada município e ano, calcula-se a média móvel dos três anos anteriores:

```text
media_movel_3anos = média do rendimento médio nos três anos anteriores
```

Depois, define-se:

```text
risco_quebra = 1 se rendimento_atual < 0,80 × media_movel_3anos
risco_quebra = 0 caso contrário
```

Essa definição evita comparar municípios diferentes diretamente, pois cada município é comparado contra seu próprio histórico.

## Variáveis explicativas

### Históricas e agrícolas

- Área plantada.
- Produtividade do ano anterior.
- Média móvel dos três anos anteriores.

### Climáticas

- Chuva total anual.
- Chuva média diária.
- Temperatura média.
- Temperatura máxima média.
- Temperatura mínima média.
- Umidade relativa média.
- Velocidade média do vento.
- Número de dias secos.
- Número de dias com chuva extrema.
- Número de dias de calor extremo.

## Validação

O projeto usa divisão temporal:

```text
Treino: anos até 2020
Teste: anos após 2020
```

Essa abordagem é mais adequada do que uma divisão aleatória simples, pois o objetivo é avaliar se o modelo treinado em anos anteriores generaliza para anos mais recentes.

## Modelos

- Regressão Logística.
- Random Forest.
- Gradient Boosting.

## Métricas

- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1-score.
- ROC AUC.
- Matriz de confusão.

A métrica prioritária é o F1-score, pois o problema pode ter desbalanceamento entre anos com e sem quebra.

## Interpretabilidade

A interpretabilidade é feita por:

- Importância interna das variáveis em modelos baseados em árvores.
- Importância por permutação, usando F1-score como métrica de degradação.

## Limitações previstas

- A agregação climática anual pode ocultar efeitos específicos da safra.
- A coordenada municipal é uma aproximação espacial.
- O modelo não inclui variáveis de solo, manejo, cultivar, tecnologia ou irrigação.
- O estudo é preditivo/classificatório, não causal.
