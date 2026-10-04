"""
Lógica de jogo do Jogo da Velha (Tic-Tac-Toe).

Todas as variáveis/funções carregam o código 32 exigido pelo enunciado
do trabalho T1 (disciplina de IA, PUCRS).

Este módulo é a "fonte da verdade" (ground truth) usada para:
  1. Rotular corretamente cada tabuleiro do dataset (o dataset original
     da UCI só informa se X venceu ou não — aqui recomputamos o estado
     real: Tem jogo, X venceu, O venceu, Empate).
  2. Gerar tabuleiros sintéticos de "Tem jogo" (estados não-terminais),
     que NÃO existem no dataset da UCI (ele só contém tabuleiros de
     final de jogo).
  3. Validar se um tabuleiro é fisicamente possível (nº de X - nº de O
     deve ser 0 ou 1, já que X sempre começa).
"""

import random
from itertools import product

CASAS_32 = 9
LINHAS_VENCEDORAS_32 = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # colunas
    (0, 4, 8), (2, 4, 6),              # diagonais
]

TEM_JOGO_32 = "tem_jogo"
X_VENCEU_32 = "x_venceu"
O_VENCEU_32 = "o_venceu"
EMPATE_32 = "empate"


def checa_vencedor_32(tabuleiro_32, simbolo_32):
    """Retorna True se `simbolo_32` ('x' ou 'o') tem 3 em linha em tabuleiro_32."""
    for a_32, b_32, c_32 in LINHAS_VENCEDORAS_32:
        if tabuleiro_32[a_32] == tabuleiro_32[b_32] == tabuleiro_32[c_32] == simbolo_32:
            return True
    return False


def tabuleiro_fisicamente_valido_32(tabuleiro_32):
    """Verifica se a contagem de x/o é compatível com um jogo real (x joga primeiro)."""
    qtd_x_32 = tabuleiro_32.count("x")
    qtd_o_32 = tabuleiro_32.count("o")
    diff_32 = qtd_x_32 - qtd_o_32
    if diff_32 not in (0, 1):
        return False
    # não pode haver dois vencedores simultâneos
    x_venceu_32 = checa_vencedor_32(tabuleiro_32, "x")
    o_venceu_32 = checa_vencedor_32(tabuleiro_32, "o")
    if x_venceu_32 and o_venceu_32:
        return False
    # se O venceu, as contagens de x e o devem ser iguais (X jogou por último só quando X vence)
    if o_venceu_32 and diff_32 != 0:
        return False
    if x_venceu_32 and diff_32 != 1:
        return False
    return True


def classifica_estado_32(tabuleiro_32):
    """
    Classifica um tabuleiro (lista/tupla de 9 chars 'x'/'o'/'b') em uma das
    4 classes do problema: tem_jogo, x_venceu, o_venceu, empate.
    Levanta ValueError se o tabuleiro não for fisicamente válido.
    """
    if not tabuleiro_fisicamente_valido_32(tabuleiro_32):
        raise ValueError(f"Tabuleiro fisicamente inválido: {tabuleiro_32}")

    x_venceu_32 = checa_vencedor_32(tabuleiro_32, "x")
    o_venceu_32 = checa_vencedor_32(tabuleiro_32, "o")
    casas_vazias_32 = tabuleiro_32.count("b")

    if x_venceu_32:
        return X_VENCEU_32
    if o_venceu_32:
        return O_VENCEU_32
    if casas_vazias_32 == 0:
        return EMPATE_32
    return TEM_JOGO_32


def jogador_da_vez_32(tabuleiro_32):
    """Retorna 'x' ou 'o': de quem é a vez de jogar."""
    qtd_x_32 = tabuleiro_32.count("x")
    qtd_o_32 = tabuleiro_32.count("o")
    return "o" if qtd_x_32 > qtd_o_32 else "x"


def gera_tabuleiro_tem_jogo_aleatorio_32(rng_32, n_min_jogadas_32=0, n_max_jogadas_32=7):
    """
    Gera, por simulação de partida aleatória, um tabuleiro em estado
    'tem_jogo' (não-terminal). Simula jogadas alternadas de x/o em casas
    aleatórias e para em um ponto aleatório da partida, desde que ainda
    não tenha vencedor nem tabuleiro cheio.
    """
    for _tentativa_32 in range(200):  # tentativas
        tabuleiro_32 = ["b"] * CASAS_32
        casas_livres_32 = list(range(CASAS_32))
        random.Random(rng_32.random()).shuffle(casas_livres_32)
        n_jogadas_alvo_32 = rng_32.randint(n_min_jogadas_32, n_max_jogadas_32)
        jogador_32 = "x"
        jogadas_feitas_32 = 0
        for pos_32 in casas_livres_32:
            if jogadas_feitas_32 >= n_jogadas_alvo_32:
                break
            tabuleiro_32[pos_32] = jogador_32
            if checa_vencedor_32(tabuleiro_32, jogador_32):
                tabuleiro_32[pos_32] = "b"  # desfaz, não queremos terminar o jogo aqui
                break
            jogador_32 = "o" if jogador_32 == "x" else "x"
            jogadas_feitas_32 += 1
        estado_32 = classifica_estado_32(tabuleiro_32)
        if estado_32 == TEM_JOGO_32:
            return tuple(tabuleiro_32)
    raise RuntimeError("Não foi possível gerar tabuleiro 'tem_jogo' após várias tentativas")


def jogada_aleatoria_32(tabuleiro_32, rng_32=random):
    """Escolhe uma casa vazia aleatória para o jogador da vez. Retorna novo tabuleiro (lista)."""
    casas_vazias_32 = [i_32 for i_32, v_32 in enumerate(tabuleiro_32) if v_32 == "b"]
    if not casas_vazias_32:
        return None
    pos_32 = rng_32.choice(casas_vazias_32)
    novo_32 = list(tabuleiro_32)
    novo_32[pos_32] = jogador_da_vez_32(tabuleiro_32)
    return novo_32


def _coord_para_indice_32(r_32, c_32):
    return r_32 * 3 + c_32


def _indice_para_coord_32(i_32):
    return divmod(i_32, 3)


def rotaciona_90_32(tabuleiro_32):
    """Rotaciona o tabuleiro 90 graus no sentido horário."""
    novo_32 = [None] * 9
    for i_32 in range(9):
        r_32, c_32 = _indice_para_coord_32(i_32)
        novo_r_32, novo_c_32 = c_32, 2 - r_32
        novo_32[_coord_para_indice_32(novo_r_32, novo_c_32)] = tabuleiro_32[i_32]
    return tuple(novo_32)


def espelha_horizontal_32(tabuleiro_32):
    """Espelha o tabuleiro na horizontal (esquerda-direita)."""
    novo_32 = [None] * 9
    for i_32 in range(9):
        r_32, c_32 = _indice_para_coord_32(i_32)
        novo_32[_coord_para_indice_32(r_32, 2 - c_32)] = tabuleiro_32[i_32]
    return tuple(novo_32)


def gera_simetrias_32(tabuleiro_32):
    """
    Retorna o conjunto (sem repetição) das 8 simetrias do grupo diedral
    D4 de um tabuleiro: 4 rotações x {original, espelhado}.
    Usado para AUMENTAR (data augmentation) classes raras (ex.: empates),
    já que uma simetria de um tabuleiro terminal válido também é um
    tabuleiro terminal válido com o mesmo rótulo de classe.
    """
    simetrias_32 = set()
    atual_32 = tuple(tabuleiro_32)
    for _rotacao_32 in range(4):
        atual_32 = rotaciona_90_32(atual_32)
        simetrias_32.add(atual_32)
        simetrias_32.add(espelha_horizontal_32(atual_32))
    return simetrias_32


def tabuleiro_para_texto_32(tabuleiro_32):
    """Representação visual 3x3 de um tabuleiro para o front end em modo texto."""
    simb_32 = {"x": "X", "o": "O", "b": " "}
    linhas_32 = []
    for i_32 in range(0, 9, 3):
        linha_32 = " | ".join(simb_32[c_32] for c_32 in tabuleiro_32[i_32:i_32 + 3])
        linhas_32.append(linha_32)
    return "\n---------\n".join(linhas_32)
