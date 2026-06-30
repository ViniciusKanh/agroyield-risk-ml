# AgroYield Risk ML

**Classificação interpretável de risco de quebra de produtividade agrícola usando dados públicos, clima e aprendizado de máquina.**

Este projeto desenvolve um pipeline experimental em Python para classificar o **risco de quebra de produtividade da soja em municípios brasileiros**, com foco inicial nos municípios do **Paraná**. A proposta combina dados agrícolas públicos da **Produção Agrícola Municipal — PAM/IBGE** com variáveis climáticas obtidas pela **NASA POWER Daily API**, utilizando técnicas de **Machine Learning supervisionado** e **interpretabilidade de modelos**.

O projeto foi estruturado para apoiar a construção de um **resumo expandido científico para o CONICT**, com ênfase em metodologia reprodutível, dados abertos, análise temporal e explicação dos fatores associados à queda de produtividade agrícola.

---

## 1. Visão geral

A agricultura brasileira está diretamente exposta à variabilidade climática, especialmente em culturas temporárias como soja, milho e feijão. Alterações no regime de chuvas, aumento da temperatura, ocorrência de estiagens e eventos extremos podem comprometer a produtividade agrícola municipal.

Neste contexto, este projeto busca responder à seguinte questão:

> É possível classificar anos de risco de quebra de produtividade agrícola em municípios produtores de soja a partir de dados históricos de produção e variáveis climáticas?

A abordagem proposta utiliza aprendizado de máquina para identificar padrões associados à queda de rendimento agrícola, mantendo foco em modelos interpretáveis e em resultados adequados para discussão científica.

---

## 2. Objetivo do projeto

### Objetivo geral

Desenvolver um pipeline de aprendizado de máquina interpretável para classificar o risco de quebra de produtividade da soja em municípios do Paraná, utilizando dados agrícolas públicos e variáveis climáticas históricas.

### Objetivos específicos

* Coletar dados agrícolas municipais da PAM/IBGE.
* Coletar dados climáticos históricos por município usando a NASA POWER Daily API.
* Integrar dados agrícolas, climáticos e geográficos em uma base única.
* Construir uma variável-alvo baseada em queda relativa de produtividade.
* Treinar modelos supervisionados para classificação de risco.
* Avaliar o desempenho dos modelos com métricas adequadas.
* Interpretar as variáveis mais relevantes para a classificação.
* Gerar tabelas, gráficos e resultados para apoio à escrita científica.

---

## 3. Problema de pesquisa

A unidade de análise do projeto é:

```text
município + ano
```

Cada linha do dataset final representa o comportamento produtivo e climático de um município em determinado ano.

A variável-alvo principal é definida da seguinte forma:

```text
risco_quebra = 1 se rendimento_medio_kg_ha < 80% da média móvel dos 3 anos anteriores

risco_quebra = 0 caso contrário
```

Essa regra permite identificar anos em que a produtividade do município ficou significativamente abaixo do seu próprio padrão histórico recente.

---

## 4. Hipótese científica

A hipótese central do projeto é:

> Municípios-ano com menor precipitação acumulada, maior número de dias secos e maior exposição a temperaturas elevadas apresentam maior probabilidade de queda relevante na produtividade da soja.

A hipótese será testada por meio de modelos supervisionados de classificação e análise de importância das variáveis.

---

## 5. Bases de dados utilizadas

### 5.1 IBGE/PAM — Produção Agrícola Municipal

A **Produção Agrícola Municipal — PAM**, disponibilizada pelo IBGE via SIDRA, fornece dados anuais sobre lavouras temporárias e permanentes nos municípios brasileiros.

Neste projeto, a PAM é usada para obter informações sobre a cultura da soja.

Link da tabela:

```text
https://sidra.ibge.gov.br/tabela/1612
```

Principais variáveis utilizadas:

| Variável             | Descrição                             |
| -------------------- | ------------------------------------- |
| Município            | Município produtor                    |
| Código do município  | Identificador territorial do IBGE     |
| Ano                  | Ano de referência                     |
| Produto agrícola     | Cultura analisada, inicialmente soja  |
| Área plantada        | Área plantada em hectares             |
| Área colhida         | Área efetivamente colhida em hectares |
| Quantidade produzida | Produção agrícola total               |
| Rendimento médio     | Produtividade média em kg/ha          |
| Valor da produção    | Valor monetário da produção           |

---

### 5.2 NASA POWER Daily API

A **NASA POWER Daily API** fornece dados meteorológicos e climáticos diários a partir de coordenadas geográficas.

Link da documentação:

```text
https://power.larc.nasa.gov/docs/services/api/temporal/daily/
```

Variáveis climáticas previstas no projeto:

| Variável    | Descrição                     |
| ----------- | ----------------------------- |
| PRECTOTCORR | Precipitação diária corrigida |
| T2M         | Temperatura média a 2 metros  |
| T2M_MAX     | Temperatura máxima            |
| T2M_MIN     | Temperatura mínima            |
| RH2M        | Umidade relativa              |
| WS2M        | Velocidade do vento           |

A partir dos dados diários, são calculadas variáveis agregadas por município e ano, como:

| Variável derivada        | Cálculo                                              |
| ------------------------ | ---------------------------------------------------- |
| Chuva total anual        | Soma da precipitação diária                          |
| Temperatura média anual  | Média da temperatura diária                          |
| Temperatura máxima média | Média das temperaturas máximas                       |
| Temperatura mínima média | Média das temperaturas mínimas                       |
| Dias secos               | Número de dias com precipitação inferior a 1 mm      |
| Dias de calor extremo    | Número de dias com temperatura máxima acima de 35 °C |

---

### 5.3 Coordenadas municipais

Para consultar a NASA POWER, cada município precisa estar associado a uma latitude e longitude.

Como solução operacional inicial, o projeto utiliza o repositório público:

```text
https://github.com/kelvins/municipios-brasileiros
```

Esse repositório fornece dados municipais com código IBGE, nome do município, estado, latitude e longitude.

### Observação metodológica

Para uma versão científica mais robusta, é possível substituir essa base auxiliar pelo cálculo dos centroides a partir da **Malha Municipal do IBGE**.

---

## 6. Recorte inicial do estudo

O recorte inicial foi definido para manter o projeto viável, objetivo e adequado ao formato de resumo expandido.

| Elemento           | Definição                         |
| ------------------ | --------------------------------- |
| Cultura agrícola   | Soja                              |
| Estado             | Paraná                            |
| Período            | 2006 a 2024                       |
| Unidade de análise | Município-ano                     |
| Tipo de problema   | Classificação supervisionada      |
| Variável-alvo      | Risco de quebra de produtividade  |
| Linguagem          | Python                            |
| Ambiente           | VSCode ou terminal local          |
| Reprodutibilidade  | Pipeline automatizado por scripts |

---

## 7. Estrutura do projeto

```text
agroyield-risk-ml/
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── dados brutos baixados das fontes originais
│   │
│   ├── interim/
│   │   └── dados intermediários tratados parcialmente
│   │
│   └── processed/
│       └── dataset final pronto para modelagem
│
├── docs/
│   ├── fontes_de_dados.md
│   ├── metodologia.md
│   ├── checklist_conict.md
│   └── roteiro_execucao_vscode.md
│
├── models/
│   └── modelos treinados em formato joblib
│
├── outputs/
│   ├── figures/
│   │   └── gráficos gerados pelo pipeline
│   │
│   └── tables/
│       └── tabelas de métricas, previsões e importâncias
│
├── reports/
│   └── resumo dos resultados experimentais
│
├── scripts/
│   ├── 00_validar_ambiente.py
│   ├── 01_baixar_municipios.py
│   ├── 02_baixar_pam_ibge.py
│   ├── 03_baixar_clima_nasa_power.py
│   ├── 04_montar_dataset.py
│   ├── 05_modelagem.py
│   ├── 06_interpretabilidade.py
│   └── 07_gerar_resumo_resultados.py
│
├── src/
│   └── conict_agro_ml/
│       └── funções reutilizáveis do projeto
│
├── requirements.txt
├── requirements-optional.txt
├── run_pipeline.py
└── README.md
```

---

## 8. Tecnologias utilizadas

| Tecnologia   | Uso                                |
| ------------ | ---------------------------------- |
| Python       | Linguagem principal                |
| Pandas       | Manipulação e tratamento dos dados |
| NumPy        | Operações numéricas                |
| Scikit-learn | Modelagem, métricas e validação    |
| Matplotlib   | Visualização dos resultados        |
| Seaborn      | Visualizações estatísticas         |
| Requests     | Consumo de APIs                    |
| PyYAML       | Leitura de configurações           |
| Joblib       | Salvamento de modelos              |
| XGBoost      | Modelo supervisionado opcional     |
| SHAP         | Interpretabilidade opcional        |

---

## 9. Como executar o projeto

### 9.1 Clonar o repositório

```bash
git clone https://github.com/seu-usuario/agroyield-risk-ml.git
cd agroyield-risk-ml
```

---

### 9.2 Criar ambiente virtual

No Windows:

```bash
python -m venv .venv
```

Ativar no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 9.3 Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para instalar bibliotecas opcionais:

```bash
pip install -r requirements-optional.txt
```

---

### 9.4 Validar o ambiente

```bash
python scripts/00_validar_ambiente.py
```

Esse script verifica se as principais bibliotecas estão instaladas corretamente e se a estrutura do projeto está disponível.

---

### 9.5 Executar o pipeline completo

```bash
python run_pipeline.py
```

Esse comando executa todas as etapas principais:

```text
1. Validação do ambiente
2. Download dos municípios
3. Download dos dados agrícolas
4. Download dos dados climáticos
5. Montagem do dataset final
6. Treinamento dos modelos
7. Interpretabilidade
8. Geração do resumo experimental
```

---

### 9.6 Executar etapa por etapa

Caso deseje acompanhar cada fase separadamente:

```bash
python scripts/01_baixar_municipios.py
python scripts/02_baixar_pam_ibge.py
python scripts/03_baixar_clima_nasa_power.py
python scripts/04_montar_dataset.py
python scripts/05_modelagem.py
python scripts/06_interpretabilidade.py
python scripts/07_gerar_resumo_resultados.py
```

---

## 10. Execução rápida para teste

Para testar o pipeline climático sem consultar todos os municípios:

```bash
python scripts/03_baixar_clima_nasa_power.py --limite-municipios 10
```

Essa opção é útil para validar a estrutura antes de executar a coleta completa.

---

## 11. Dataset final esperado

Ao final da etapa de integração, o projeto deve gerar o arquivo:

```text
data/processed/dataset_agro_ml.csv
```

Estrutura esperada do dataset:

| Coluna                     | Origem             |
| -------------------------- | ------------------ |
| codigo_municipio           | IBGE/PAM           |
| municipio                  | IBGE/PAM           |
| uf                         | Base de municípios |
| ano                        | IBGE/PAM           |
| cultura                    | IBGE/PAM           |
| area_plantada_ha           | IBGE/PAM           |
| area_colhida_ha            | IBGE/PAM           |
| quantidade_produzida_t     | IBGE/PAM           |
| rendimento_medio_kg_ha     | IBGE/PAM           |
| valor_producao_mil_reais   | IBGE/PAM           |
| latitude                   | Base municipal     |
| longitude                  | Base municipal     |
| chuva_total_mm             | NASA POWER         |
| temperatura_media_c        | NASA POWER         |
| temperatura_maxima_media_c | NASA POWER         |
| temperatura_minima_media_c | NASA POWER         |
| umidade_media              | NASA POWER         |
| vento_medio                | NASA POWER         |
| dias_secos                 | Calculado          |
| dias_calor_extremo         | Calculado          |
| media_movel_3anos          | Calculado          |
| queda_percentual           | Calculado          |
| risco_quebra               | Calculado          |

---

## 12. Modelagem

O projeto utiliza modelos supervisionados para classificar a variável `risco_quebra`.

Modelos previstos:

| Modelo              | Finalidade                             |
| ------------------- | -------------------------------------- |
| Regressão Logística | Modelo de referência simples           |
| Árvore de Decisão   | Modelo interpretável                   |
| Random Forest       | Modelo robusto principal               |
| XGBoost             | Modelo comparativo de maior desempenho |

---

## 13. Validação experimental

A divisão dos dados deve respeitar a ordem temporal, evitando que informações futuras sejam usadas para treinar o modelo.

Estratégia recomendada:

```text
Treino: anos mais antigos
Teste: anos mais recentes
```

Exemplo:

```text
Treino: 2006 a 2019
Teste: 2020 a 2024
```

Essa abordagem é mais adequada para problemas temporais do que uma divisão aleatória simples.

---

## 14. Métricas de avaliação

As métricas usadas para avaliar os modelos são:

| Métrica            | Interpretação                                                     |
| ------------------ | ----------------------------------------------------------------- |
| Accuracy           | Percentual geral de acertos                                       |
| Precision          | Proporção de alertas corretos entre os casos previstos como risco |
| Recall             | Capacidade de detectar casos reais de quebra                      |
| F1-score           | Equilíbrio entre precision e recall                               |
| AUC-ROC            | Capacidade geral de separação entre as classes                    |
| Matriz de confusão | Distribuição dos acertos e erros do modelo                        |

Como a classe de risco pode ser desbalanceada, o **F1-score** e o **recall** devem receber atenção especial.

---

## 15. Interpretabilidade

A interpretabilidade é parte central do projeto. O objetivo não é apenas prever, mas também compreender quais fatores estão mais associados ao risco de quebra.

Métodos previstos:

| Técnica                              | Uso                                                     |
| ------------------------------------ | ------------------------------------------------------- |
| Importância interna do Random Forest | Identificar variáveis mais usadas pelo modelo           |
| Importância por permutação           | Avaliar impacto das variáveis na performance            |
| SHAP                                 | Explicação local e global das previsões, caso instalado |

Exemplos de perguntas interpretáveis:

* A chuva acumulada influencia a classificação de risco?
* O número de dias secos aumenta a probabilidade de quebra?
* Temperaturas máximas elevadas estão associadas à queda de produtividade?
* Variáveis agrícolas históricas são mais relevantes do que variáveis climáticas?

---

## 16. Resultados esperados

Ao final da execução, espera-se gerar os seguintes artefatos:

```text
data/processed/dataset_agro_ml.csv

outputs/tables/metricas_modelos.csv
outputs/tables/importancia_variaveis.csv
outputs/tables/previsoes_teste.csv

outputs/figures/matriz_confusao_*.png
outputs/figures/importancia_variaveis_*.png

reports/resumo_resultados.md

models/melhor_modelo.joblib
```

---

## 17. Resultados científicos esperados

O projeto deve permitir discutir:

* Se modelos de Machine Learning conseguem classificar anos de risco de quebra de produtividade.
* Quais variáveis climáticas mais influenciam a classificação.
* Se a combinação de dados agrícolas e climáticos melhora a análise.
* Se a abordagem pode apoiar monitoramento agrícola, planejamento rural ou políticas públicas.
* Quais limitações existem ao usar dados agregados em escala municipal.

---

## 18. Cuidados metodológicos

Este projeto adota alguns cuidados para reduzir erros comuns em estudos com Machine Learning aplicado:

* A divisão treino-teste respeita a ordem temporal.
* A produtividade atual não deve ser usada como variável explicativa direta.
* A variável-alvo é construída a partir do histórico municipal.
* As variáveis climáticas são agregadas antes da modelagem.
* Os resultados devem ser analisados com métricas além da acurácia.
* A interpretação dos modelos não deve ser tratada como causalidade.
* As conclusões devem ser apresentadas como associação estatística, não como prova causal.

---

## 19. Limitações

Algumas limitações devem ser consideradas:

* A análise usa dados agregados por município e ano.
* A produtividade agrícola pode ser influenciada por fatores não incluídos, como manejo, solo, tecnologia, pragas, cultivares, crédito rural e práticas locais.
* A NASA POWER fornece dados estimados por coordenadas, não medições diretas em cada propriedade rural.
* A definição de quebra baseada em média móvel é operacional e pode ser ajustada em estudos futuros.
* A associação entre clima e produtividade não implica causalidade direta.

---

## 20. Possíveis melhorias futuras

Possíveis extensões do projeto:

* Comparar soja, milho e feijão.
* Expandir a análise para outros estados brasileiros.
* Utilizar estações do INMET como fonte climática alternativa.
* Calcular centroides a partir da Malha Municipal do IBGE.
* Incluir variáveis de uso e cobertura da terra do MapBiomas.
* Testar janelas climáticas por safra em vez de ano civil.
* Usar validação temporal com múltiplas janelas.
* Aplicar modelos explicáveis adicionais, como Explainable Boosting Machine.
* Construir um painel interativo com os resultados por município.

---

## 21. Relação com o CONICT

Este projeto foi planejado para servir como base experimental de um resumo expandido para o CONICT.

A estrutura científica sugerida para o texto final é:

```text
1. Introdução
2. Objetivo
3. Materiais e Métodos
4. Resultados e Discussão
5. Considerações Finais
6. Referências
```

Como o resumo expandido do CONICT é enxuto, o projeto computacional deve gerar evidências objetivas, como:

* Tabela de métricas dos modelos.
* Matriz de confusão.
* Gráfico de importância das variáveis.
* Breve análise interpretativa dos principais fatores associados ao risco.

---

## 22. Reprodutibilidade

O projeto busca seguir boas práticas de reprodutibilidade:

* Separação entre dados brutos, intermediários e processados.
* Scripts numerados por etapa.
* Configurações centralizadas em `config.yaml`.
* Salvamento dos modelos treinados.
* Exportação automática de tabelas e figuras.
* Documentação das fontes de dados.
* Pipeline executável por comando único.

---

## 23. Licença

Este projeto pode ser distribuído sob a licença MIT.

Sugestão de arquivo `LICENSE`:

```text
MIT License

Copyright (c) 2026 Vinicius de Souza Santos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the conditions of the MIT License.
```

---

## 24. Créditos

### Autor

**Vinicius de Souza Santos**
Estudante de Engenharia da Computação no IFSP — Campus Birigui
Pesquisador em formação nas áreas de Ciência de Dados, Machine Learning, Desenvolvimento de Software e aplicações computacionais para problemas reais.

### Desenvolvimento científico e computacional

Projeto desenvolvido como estudo experimental em aprendizado de máquina aplicado ao agronegócio, com foco em dados públicos, análise temporal, classificação supervisionada e interpretabilidade de modelos.

### Fontes de dados

* **IBGE — Instituto Brasileiro de Geografia e Estatística**
  Produção Agrícola Municipal — PAM/SIDRA.

* **NASA POWER Project**
  Dados climáticos e meteorológicos diários obtidos por latitude e longitude.

* **Repositório `kelvins/municipios-brasileiros`**
  Base auxiliar de coordenadas municipais brasileiras.

### Apoio ferramental

* Python
* Pandas
* Scikit-learn
* NASA POWER API
* SIDRA/IBGE
* VSCode
* GitHub

---

## 27. Aviso científico

Este projeto não tem o objetivo de substituir análises agronômicas especializadas. Os resultados devem ser interpretados como evidências estatísticas e computacionais exploratórias, baseadas em dados públicos agregados.

A classificação de risco gerada pelos modelos representa uma estimativa computacional e deve ser analisada com cautela, considerando limitações dos dados, escala territorial e ausência de variáveis agronômicas locais.

---

