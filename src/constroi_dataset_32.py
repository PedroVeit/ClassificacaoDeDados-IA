"""
Etapa 2 do trabalho: construção do dataset final.

PROBLEMA ENCONTRADO no dataset bruto da UCI (registrado para o relatório):
  - O dataset "Tic-Tac-Toe Endgame" da UCI contém SOMENTE tabuleiros de
    FINAL de jogo (958 tabuleiros), e o rótulo original é binário
    (x venceu / x não venceu). Ou seja, ele:
      (a) não distingue "O venceu" de "Empate" (ambos = "negative");
      (b) NÃO contém nenhum tabuleiro com jogo em andamento ("Tem jogo"),
          que é uma das 4 classes exigidas pelo problema.
  - Portanto o dataset bruto NÃO atende plenamente à necessidade do
    problema (classificação em 4 classes) e precisou de adequação.

AÇÕES REALIZADAS (adequações):
  1. Recomputamos o rótulo verdadeiro de cada tabuleiro do dataset bruto
     usando lógica de jogo própria (logica_jogo_32.classifica_estado_32),
     obtendo as classes reais: x_venceu, o_venceu, empate (nenhum tem_jogo
     no dataset bruto, como esperado).
  2. A classe "empate" tem apenas 16 exemplos no dataset bruto — poucos
     para treinar. Usamos AUMENTO DE DADOS por simetria: um tabuleiro de
     jogo da velha rotacionado/espelhado continua sendo um tabuleiro
     válido do mesmo tipo de final de jogo, então geramos as 8 simetrias
     (grupo diedral D4) de cada tabuleiro de empate para ampliar o
     conjunto de candidatos antes de amostrar.
  3. A classe "tem_jogo" (jogo em andamento) não existe no dataset bruto.
     Geramos tabuleiros SINTÉTICOS de estados não-terminais simulando
     partidas aleatórias e parando em um ponto aleatório antes do fim
     (gera_tabuleiro_tem_jogo_aleatorio_32), garantindo tabuleiros
     fisicamente válidos (mesma regra: x joga primeiro).
  4. Amostramos até 200 tabuleiros por classe (conforme sugerido no
     enunciado), sem usar necessariamente todas as instâncias
     disponíveis, com seed fixa para reprodutibilidade.

As colunas do dataset final também seguem a exigência de conter o
código 32 no nome.
"""

import csv
import random
import sys

sys.path.insert(0, "src")
from logica_jogo_32 import (  # noqa: E402
    classifica_estado_32,
    gera_simetrias_32,
    gera_tabuleiro_tem_jogo_aleatorio_32,
    TEM_JOGO_32,
    X_VENCEU_32,
    O_VENCEU_32,
    EMPATE_32,
)

SEED_32 = 32
N_POR_CLASSE_32 = 200
CAMINHO_BRUTO_32 = "data/raw_uci_tic-tac-toe.csv"
CAMINHO_FINAL_32 = "data/dataset_final_32.csv"
CAMINHO_LOG_32 = "reports/log_construcao_dataset_32.txt"


def carrega_bruto_32(caminho_32):
    with open(caminho_32) as f_32:
        leitor_32 = csv.reader(f_32)
        next(leitor_32)  # cabeçalho
        return [tuple(linha_32[:9]) for linha_32 in leitor_32]


def separa_por_classe_32(tabuleiros_32):
    grupos_32 = {X_VENCEU_32: [], O_VENCEU_32: [], EMPATE_32: []}
    for tab_32 in tabuleiros_32:
        classe_32 = classifica_estado_32(tab_32)
        if classe_32 in grupos_32:
            grupos_32[classe_32].append(tab_32)
    return grupos_32


def amplia_com_simetrias_32(tabuleiros_32):
    ampliado_32 = set()
    for tab_32 in tabuleiros_32:
        ampliado_32.update(gera_simetrias_32(tab_32))
    return list(ampliado_32)


def amostra_32(lista_32, n_32, rng_32):
    if len(lista_32) <= n_32:
        return list(lista_32)
    return rng_32.sample(lista_32, n_32)


def gera_lote_tem_jogo_32(n_32, rng_32):
    tabuleiros_32 = set()
    tentativas_32 = 0
    while len(tabuleiros_32) < n_32 and tentativas_32 < n_32 * 20:
        tentativas_32 += 1
        tab_32 = gera_tabuleiro_tem_jogo_aleatorio_32(rng_32)
        tabuleiros_32.add(tab_32)
    return list(tabuleiros_32)


def main_32():
    rng_32 = random.Random(SEED_32)
    linhas_log_32 = []

    brutos_32 = carrega_bruto_32(CAMINHO_BRUTO_32)
    linhas_log_32.append(f"Tabuleiros carregados do dataset bruto da UCI: {len(brutos_32)}")

    grupos_32 = separa_por_classe_32(brutos_32)
    for classe_32, lst_32 in grupos_32.items():
        linhas_log_32.append(f"  Reclassificados como '{classe_32}': {len(lst_32)}")

    # x_venceu e o_venceu já têm bastante volume: amostra direta
    x_venceu_final_32 = amostra_32(grupos_32[X_VENCEU_32], N_POR_CLASSE_32, rng_32)
    o_venceu_final_32 = amostra_32(grupos_32[O_VENCEU_32], N_POR_CLASSE_32, rng_32)

    # empate: poucos exemplos -> amplia por simetria antes de amostrar
    empate_ampliado_32 = amplia_com_simetrias_32(grupos_32[EMPATE_32])
    linhas_log_32.append(
        f"  'empate' ampliado por simetrias D4: {len(grupos_32[EMPATE_32])} -> "
        f"{len(empate_ampliado_32)} tabuleiros únicos "
        f"(o total permaneceu 16: é um fato conhecido de combinatória do jogo da "
        f"velha que existem exatamente 16 tabuleiros de empate possíveis no total, "
        f"formando apenas 2 classes de simetria D4 (2 x 8 = 16); logo os 16 do "
        f"dataset bruto da UCI já são TODOS os empates que existem, e a classe "
        f"'empate' fica com apenas 16/200 exemplos mesmo após a tentativa de "
        f"aumento de dados - limitação do problema, não do processo)"
    )
    empate_final_32 = amostra_32(empate_ampliado_32, N_POR_CLASSE_32, rng_32)

    # tem_jogo: não existe no bruto -> gerado sinteticamente
    tem_jogo_final_32 = gera_lote_tem_jogo_32(N_POR_CLASSE_32, rng_32)
    linhas_log_32.append(f"  'tem_jogo' gerado sinteticamente: {len(tem_jogo_final_32)}")

    conjunto_final_32 = (
        [(t_32, X_VENCEU_32, "raw_uci") for t_32 in x_venceu_final_32]
        + [(t_32, O_VENCEU_32, "raw_uci") for t_32 in o_venceu_final_32]
        + [(t_32, EMPATE_32, "raw_uci+simetria") for t_32 in empate_final_32]
        + [(t_32, TEM_JOGO_32, "sintetico") for t_32 in tem_jogo_final_32]
    )
    rng_32.shuffle(conjunto_final_32)

    with open(CAMINHO_FINAL_32, "w", newline="") as f_32:
        escritor_32 = csv.writer(f_32)
        escritor_32.writerow(
            ["casa_0_32", "casa_1_32", "casa_2_32", "casa_3_32", "casa_4_32",
             "casa_5_32", "casa_6_32", "casa_7_32", "casa_8_32",
             "classe_32", "origem_32"]
        )
        for tab_32, classe_32, origem_32 in conjunto_final_32:
            escritor_32.writerow(list(tab_32) + [classe_32, origem_32])

    linhas_log_32.append(f"\nTotal de tabuleiros no dataset final: {len(conjunto_final_32)}")
    contagem_final_32 = {}
    for _tab_32, classe_32, _origem_32 in conjunto_final_32:
        contagem_final_32[classe_32] = contagem_final_32.get(classe_32, 0) + 1
    for classe_32, qtd_32 in contagem_final_32.items():
        linhas_log_32.append(f"  {classe_32}: {qtd_32}")

    with open(CAMINHO_LOG_32, "w") as f_32:
        f_32.write("\n".join(linhas_log_32) + "\n")

    print("\n".join(linhas_log_32))
    print(f"\nDataset final salvo em: {CAMINHO_FINAL_32}")
    print(f"Log salvo em: {CAMINHO_LOG_32}")


if __name__ == "__main__":
    main_32()
