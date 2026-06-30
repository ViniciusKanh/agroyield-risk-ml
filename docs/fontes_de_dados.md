# Fontes de dados

## 1. IBGE/PAM — SIDRA Tabela 1612

Link principal: https://sidra.ibge.gov.br/tabela/1612

Uso no projeto:

- Área plantada.
- Área colhida.
- Quantidade produzida.
- Rendimento médio da produção.
- Valor da produção.

Recorte configurado:

- Tabela: 1612.
- Produto: Soja (em grão).
- Código do produto no SIDRA usado no projeto: 2713.
- Classificação do produto: 81.
- Nível territorial: município.
- UF: Paraná, código 41.

Exemplo conceitual de URL da API:

```text
https://apisidra.ibge.gov.br/values/t/1612/n6/in%20n3%2041/v/all/p/2006,2007,2008/c81/2713?formato=json
```

Se a API falhar, caminho manual:

1. Acesse https://sidra.ibge.gov.br/tabela/1612.
2. Selecione os anos do estudo.
3. Selecione Unidade Territorial: Municípios do Paraná.
4. Selecione Produto das lavouras temporárias: Soja (em grão).
5. Selecione todas as variáveis principais.
6. Baixe em CSV.
7. Salve em `data/raw/pam_ibge/`.

## 2. NASA POWER Daily API

Link: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

Uso no projeto:

- Precipitação diária corrigida: `PRECTOTCORR`.
- Temperatura média: `T2M`.
- Temperatura máxima: `T2M_MAX`.
- Temperatura mínima: `T2M_MIN`.
- Umidade relativa: `RH2M`.
- Velocidade do vento: `WS2M`.

Exemplo conceitual de consulta:

```text
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,WS2M&community=AG&longitude=-51.0&latitude=-23.0&start=20060101&end=20241231&format=JSON
```

## 3. Coordenadas municipais

Atalho operacional usado no MVP:

https://github.com/kelvins/municipios-brasileiros

Arquivo CSV usado pelo script:

```text
https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv
```

Alternativa científica mais forte para a versão final:

- Usar a Malha Municipal do IBGE.
- Calcular o centroide de cada município com GeoPandas.
- Usar o centroide como ponto de consulta da NASA POWER.

Link IBGE Malha Municipal:

https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html
