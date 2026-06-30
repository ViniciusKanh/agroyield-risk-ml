# Checklist para transformar o experimento em resumo expandido do CONICT

## Antes do artigo

- [ ] Rodar o pipeline completo sem erro.
- [ ] Conferir se há quantidade suficiente de municípios e anos.
- [ ] Conferir a distribuição da variável `risco_quebra`.
- [ ] Verificar se o teste temporal tem casos positivos e negativos.
- [ ] Escolher o melhor modelo pelo F1-score e balanced accuracy.
- [ ] Conferir matriz de confusão.
- [ ] Conferir importância das variáveis.
- [ ] Salvar os principais gráficos em `outputs/figures/`.
- [ ] Salvar as principais tabelas em `outputs/tables/`.

## Estrutura sugerida do resumo expandido

1. Introdução.
2. Objetivo.
3. Materiais e métodos.
4. Resultados e discussão.
5. Conclusão.
6. Referências.

## Figuras/tabelas candidatas

- Tabela 1: descrição das bases e variáveis.
- Tabela 2: comparação dos modelos.
- Figura 1: matriz de confusão do melhor modelo.
- Figura 2: importância das variáveis.

## Cuidado de escrita

Não escrever:

```text
O clima causou a quebra de produtividade.
```

Escrever:

```text
As variáveis climáticas apresentaram associação preditiva com anos classificados como risco de quebra de produtividade.
```

## Limite de páginas

Verificar o edital/modelo da edição vigente. O projeto foi estruturado para caber em texto enxuto de 5 a 6 páginas.
