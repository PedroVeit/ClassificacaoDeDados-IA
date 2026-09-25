"""
Etapa 6 do trabalho: front end mínimo (texto) do jogo da velha.

- Um jogador é humano (sempre "x", começa jogando).
- O outro jogador é a máquina, jogando de forma ALEATÓRIA ("o").
- A cada jogada (de qualquer um dos dois), a IA treinada (Árvore de
  Decisão, abordagem2 - ver salva_modelo_final_32.py) classifica o
  estado do tabuleiro e o front end mostra a mensagem correspondente.

Sobre o requisito "encerre o jogo quando a IA não detectar o fim de
jogo, e continue quando ela detectar o fim de jogo incorretamente":
  o FLUXO do jogo (perguntar próxima jogada ou não) é controlado pelo
  estado REAL do tabuleiro (calculado por logica_jogo_32, que é a
  verdade-terreno/ground truth e nunca erra), para que o jogo em si
  nunca trave. A predição da IA é mostrada a cada jogada e comparada
  com esse estado real: sempre que ela diverge, é contado como ERRO da
  IA e reportado ao usuário na hora (é exatamente o cenário descrito no
  enunciado: um "falso tem_jogo" quando o jogo já acabou, ou um "falso
  fim de jogo" quando ainda há jogo). Ao final, acertos/erros e a
  acurácia da IA durante a sessão são mostrados e also acrescentados
  a reports/log_interacoes_frontend_32.csv, para constar no relatório.
"""

import csv
import datetime
import os
import random
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from logica_jogo_32 import (  # noqa: E402
    TEM_JOGO_32, classifica_estado_32, jogada_aleatoria_32,
    jogador_da_vez_32, tabuleiro_para_texto_32,
)
from preprocessamento_32 import extrai_features_abordagem2_32  # noqa: E402

CAMINHO_MODELO_32 = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_final_32.joblib")
CAMINHO_LOG_32 = os.path.join(os.path.dirname(__file__), "..", "reports", "log_interacoes_frontend_32.csv")

MENSAGENS_32 = {
    "tem_jogo": "Tem jogo. Continue jogando!",
    "x_venceu": "X venceu!",
    "o_venceu": "O venceu!",
    "empate": "Empate!",
}


def carrega_modelo_32():
    pacote_32 = joblib.load(CAMINHO_MODELO_32)
    return pacote_32["modelo_32"], pacote_32["colunas_32"]


def prediz_estado_32(modelo_32, colunas_32, tabuleiro_32):
    feats_32 = extrai_features_abordagem2_32(tabuleiro_32)
    linha_32 = pd.DataFrame([feats_32], columns=colunas_32)
    predicao_32 = modelo_32.predict(linha_32)[0]
    return predicao_32


def pede_jogada_humana_32(tabuleiro_32):
    while True:
        entrada_32 = input("Sua jogada (posição 1-9, esquerda->direita, cima->baixo): ").strip()
        if not entrada_32.isdigit() or not (1 <= int(entrada_32) <= 9):
            print("Entrada inválida. Digite um número de 1 a 9.")
            continue
        pos_32 = int(entrada_32) - 1
        if tabuleiro_32[pos_32] != "b":
            print("Essa casa já está ocupada. Escolha outra.")
            continue
        return pos_32


def joga_partida_32(modelo_32, colunas_32, rng_32):
    tabuleiro_32 = ["b"] * 9
    acertos_ia_32 = 0
    erros_ia_32 = 0
    n_jogadas_32 = 0

    print("\nTabuleiro inicial:")
    print(tabuleiro_para_texto_32(tabuleiro_32))

    while True:
        vez_32 = jogador_da_vez_32(tabuleiro_32)
        if vez_32 == "x":
            pos_32 = pede_jogada_humana_32(tabuleiro_32)
            tabuleiro_32[pos_32] = "x"
        else:
            novo_tabuleiro_32 = jogada_aleatoria_32(tabuleiro_32, rng_32)
            tabuleiro_32 = novo_tabuleiro_32
            print("Computador (O) jogou aleatoriamente.")

        n_jogadas_32 += 1
        print("\n" + tabuleiro_para_texto_32(tabuleiro_32))

        estado_real_32 = classifica_estado_32(tuple(tabuleiro_32))
        estado_previsto_32 = prediz_estado_32(modelo_32, colunas_32, tuple(tabuleiro_32))

        acertou_32 = estado_previsto_32 == estado_real_32
        if acertou_32:
            acertos_ia_32 += 1
        else:
            erros_ia_32 += 1

        print(f"IA (código 32) diz: {MENSAGENS_32[estado_previsto_32]}"
              + ("" if acertou_32 else f"   [IA ERROU! estado real era: {MENSAGENS_32[estado_real_32]}]"))

        if estado_real_32 != TEM_JOGO_32:
            print(f"\n>>> Fim de jogo real: {MENSAGENS_32[estado_real_32]}")
            break

    print(f"\nJogadas: {n_jogadas_32} | Acertos da IA: {acertos_ia_32} | Erros da IA: {erros_ia_32} "
          f"| Acurácia da sessão: {acertos_ia_32 / n_jogadas_32:.2%}")
    return n_jogadas_32, acertos_ia_32, erros_ia_32


def registra_log_32(n_jogadas_32, acertos_32, erros_32):
    existe_32 = os.path.exists(CAMINHO_LOG_32)
    with open(CAMINHO_LOG_32, "a", newline="") as f_32:
        w_32 = csv.writer(f_32)
        if not existe_32:
            w_32.writerow(["timestamp_32", "n_jogadas_32", "acertos_ia_32", "erros_ia_32", "acuracia_32"])
        w_32.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            n_jogadas_32, acertos_32, erros_32,
            round(acertos_32 / n_jogadas_32, 4) if n_jogadas_32 else 0,
        ])


def main_32():
    modelo_32, colunas_32 = carrega_modelo_32()
    rng_32 = random.Random()
    print("=== Jogo da Velha com IA classificadora (código 32) ===")
    print("Você é X. O computador (O) joga de forma aleatória.")

    jogar_32 = "s"
    while jogar_32 == "s":
        n_jogadas_32, acertos_32, erros_32 = joga_partida_32(modelo_32, colunas_32, rng_32)
        registra_log_32(n_jogadas_32, acertos_32, erros_32)
        jogar_32 = input("\nJogar novamente? (s/n): ").strip().lower()


if __name__ == "__main__":
    main_32()
