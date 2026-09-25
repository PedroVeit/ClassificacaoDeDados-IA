"""
Etapa 3 do trabalho: pré-processamento dos dados em DUAS abordagens.

Abordagem 1 (abordagem1_32): entrada = apenas o tabuleiro atual, com os
  valores das 9 casas transformados em numéricos (b=0, x=1, o=2).
  É a transformação mais direta pedida no enunciado. Limitação registrada
  no relatório: essa codificação inteira introduz uma noção de ordem
  (0 < 1 < 2) que não existe de fato entre "vazio", "x" e "o" — os
  algoritmos usados devem ser robustos a isso (árvores lidam bem;
  kNN/MLP são mais sensíveis, o que é discutido na comparação de
  resultados).

Abordagem 2 (abordagem2_32): 7 features de mais alto nível, exigidas
  pelo enunciado:
    - qtd_x_32: quantidade de X no tabuleiro
    - qtd_o_32: quantidade de O no tabuleiro
    - qtd_ocupadas_32: posições ocupadas (qtd_x_32 + qtd_o_32)
    - linhas_2x_32: número de linhas/colunas/diagonais com exatamente 2 X e 1 vazio
    - linhas_2o_32: idem para O
    - qtd_vazias_32: número de casas vazias
    - jogador_vez_32: de quem é a vez (0 = x, 1 = o)

Ambas as abordagens usam o MESMO dataset_final_32.csv e mantêm o índice
original de cada tabuleiro (coluna id_tabuleiro_32) para garantir que a
divisão treino/validação/teste (feita depois, em divide_dataset_32.py)
seja idêntica nas duas abordagens — condição exigida pelo enunciado,
já que vários algoritmos serão comparados sobre os mesmos conjuntos.
"""

import csv

from logica_jogo_32 import LINHAS_VENCEDORAS_32, jogador_da_vez_32

CAMINHO_ENTRADA_32 = "data/dataset_final_32.csv"
CAMINHO_ABORDAGEM1_32 = "data/features_abordagem1_32.csv"
CAMINHO_ABORDAGEM2_32 = "data/features_abordagem2_32.csv"

MAPA_NUMERICO_32 = {"b": 0, "x": 1, "o": 2}


def conta_linhas_com_2_e_1_vazio_32(tabuleiro_32, simbolo_32):
    total_32 = 0
    for a_32, b_32, c_32 in LINHAS_VENCEDORAS_32:
        trio_32 = [tabuleiro_32[a_32], tabuleiro_32[b_32], tabuleiro_32[c_32]]
        if trio_32.count(simbolo_32) == 2 and trio_32.count("b") == 1:
            total_32 += 1
    return total_32


def extrai_features_abordagem1_32(tabuleiro_32):
    return [MAPA_NUMERICO_32[c_32] for c_32 in tabuleiro_32]


def extrai_features_abordagem2_32(tabuleiro_32):
    qtd_x_32 = tabuleiro_32.count("x")
    qtd_o_32 = tabuleiro_32.count("o")
    qtd_vazias_32 = tabuleiro_32.count("b")
    return [
        qtd_x_32,
        qtd_o_32,
        qtd_x_32 + qtd_o_32,
        conta_linhas_com_2_e_1_vazio_32(tabuleiro_32, "x"),
        conta_linhas_com_2_e_1_vazio_32(tabuleiro_32, "o"),
        qtd_vazias_32,
        0 if jogador_da_vez_32(tabuleiro_32) == "x" else 1,
    ]


def main_32():
    with open(CAMINHO_ENTRADA_32) as f_32:
        leitor_32 = csv.reader(f_32)
        next(leitor_32)
        linhas_32 = list(leitor_32)

    with open(CAMINHO_ABORDAGEM1_32, "w", newline="") as f1_32, \
         open(CAMINHO_ABORDAGEM2_32, "w", newline="") as f2_32:
        w1_32 = csv.writer(f1_32)
        w2_32 = csv.writer(f2_32)
        w1_32.writerow(
            ["id_tabuleiro_32"]
            + [f"casa_{i_32}_num_32" for i_32 in range(9)]
            + ["classe_32"]
        )
        w2_32.writerow(
            ["id_tabuleiro_32", "qtd_x_32", "qtd_o_32", "qtd_ocupadas_32",
             "linhas_2x_32", "linhas_2o_32", "qtd_vazias_32", "jogador_vez_32",
             "classe_32"]
        )
        for id_32, linha_32 in enumerate(linhas_32):
            tabuleiro_32 = tuple(linha_32[:9])
            classe_32 = linha_32[9]
            w1_32.writerow([id_32] + extrai_features_abordagem1_32(tabuleiro_32) + [classe_32])
            w2_32.writerow([id_32] + extrai_features_abordagem2_32(tabuleiro_32) + [classe_32])

    print(f"Abordagem 1 (tabuleiro bruto -> numérico): {CAMINHO_ABORDAGEM1_32} ({len(linhas_32)} linhas, 9 features)")
    print(f"Abordagem 2 (7 features de alto nível): {CAMINHO_ABORDAGEM2_32} ({len(linhas_32)} linhas, 7 features)")


if __name__ == "__main__":
    main_32()
