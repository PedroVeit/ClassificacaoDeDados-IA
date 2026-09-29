# T1 — Jogo da Velha com ML (IA / PUCRS — Profa. Silvia Moraes)

Classificador do estado de um tabuleiro de jogo da velha 3x3 em 4 classes:
**Tem jogo**, **X venceu**, **O venceu**, **Empate**.

> Convenção do enunciado: todas as variáveis do código carregam o código **32**.

## Estrutura do repositório

```
data/
  raw_uci_tic-tac-toe.csv       # dataset bruto da UCI (958 tabuleiros, só endgame)
  dataset_final_32.csv          # dataset final balanceado (4 classes)
  features_abordagem1_32.csv    # abordagem 1: tabuleiro -> numérico
  features_abordagem2_32.csv    # abordagem 2: 7 features de alto nível
  split_32.csv                  # divisão treino/validação/teste (compartilhada)
src/
  logica_jogo_32.py             # ground truth: regras do jogo da velha
  constroi_dataset_32.py        # Etapa 2 — monta o dataset final
  preprocessamento_32.py        # Etapa 3 — as 2 abordagens de features
  divide_dataset_32.py          # Etapa 4 — split treino/val/teste
  treina_modelos_32.py          # Etapa 5 — treina e compara os 5 algoritmos
  salva_modelo_final_32.py      # salva o modelo escolhido para o front end
  frontend_32.py                # Etapa 6 — front end em texto (jogável)
models/
  modelo_final_32.joblib        # modelo final (Árvore de Decisão, abordagem2)
reports/
  log_construcao_dataset_32.txt     # log/decisões da construção do dataset
  resultados_modelos_32.csv         # tabela com as métricas dos 10 experimentos
  matrizes_confusao_32.json         # matrizes de confusão de cada experimento
  comparacao_acuracia_32.png        # gráfico comparativo
  log_interacoes_frontend_32.csv    # histórico de partidas jogadas no front end
```

## Como rodar (na ordem)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/constroi_dataset_32.py      # gera data/dataset_final_32.csv
python3 src/preprocessamento_32.py      # gera as 2 abordagens de features
python3 src/divide_dataset_32.py        # gera o split treino/val/teste
python3 src/treina_modelos_32.py        # treina e compara os 5 algoritmos
python3 src/salva_modelo_final_32.py    # salva o modelo final para o front end
python3 src/frontend_32.py              # jogar!
```

## Resumo das decisões (detalhes completos em `reports/log_construcao_dataset_32.txt`)

- O dataset bruto da UCI só tem tabuleiros de **final de jogo** e rótulo
  binário. Precisou ser **relabelado** (com lógica de jogo própria) e
  **complementado**: a classe "Tem jogo" não existe no dataset original e
  foi **gerada sinteticamente**; a classe "Empate" tem só 16 exemplos
  possíveis **no total** (fato combinatório do jogo da velha — não é
  possível conseguir mais, mesmo com aumento por simetria).
- Dataset final: 200 / 200 / 200 / 16 exemplos (x_venceu / o_venceu /
  tem_jogo / empate).
- Duas abordagens de pré-processamento testadas; a abordagem 2 (7
  features de alto nível) generaliza muito melhor com poucos dados, mas
  tem um **teto estrutural de acurácia** porque as features não
  informam diretamente "há uma linha completa" (só "quase-linha"),
  então confunde alguns tabuleiros cheios/quase cheios. Ver observação
  detalhada no log.
- 5 algoritmos comparados (kNN, Árvore de Decisão, MLP, Random Forest,
  SVM) x 2 abordagens = 10 experimentos, com seleção de hiperparâmetros
  via conjunto de validação.
- Modelo final escolhido para o front end: **Árvore de Decisão,
  abordagem 2** (empatou em 1º lugar em acurácia, é a mais rápida e a
  mais interpretável).

## Relatório e vídeo

O relatório (formato PPT) deve reunir: introdução, dataset e suas
modificações, pré-processamento, algoritmos e parametrização,
comparação de resultados (tabela `reports/resultados_modelos_32.csv` e
gráfico `reports/comparacao_acuracia_32.png`), resultados do front end
(`reports/log_interacoes_frontend_32.csv`) e conclusão. Um esqueleto
inicial está em `reports/Relatorio_T1_32.pptx`.

**Ferramentas de IA usadas:** Claude (Anthropic) — usado para
estruturar o pipeline de dados, implementar a lógica de jogo, o
pré-processamento, o treinamento comparativo dos modelos e o front
end. *(ajuste esta linha para refletir exatamente o que sua dupla usou — o enunciado pede para declarar isso.)*
