"""
Etapa 4 do trabalho: divisão física do dataset em treino / validação / teste.

Requisito do enunciado: os MESMOS conjuntos devem ser usados em todos os
experimentos (mesmos exemplos em treino/validação/teste), pois vários
algoritmos serão comparados entre si nas duas abordagens de
pré-processamento. Por isso a divisão é feita UMA VEZ, por
id_tabuleiro_32 (chave comum às duas abordagens), e o resultado
(um rótulo de conjunto por tabuleiro) é salvo em
data/split_32.csv e depois apenas "juntado" às features de cada
abordagem nos scripts de treino.

Proporções: 60% treino / 20% validação / 20% teste, com estratificação
pela classe (classe_32) para preservar a proporção de cada classe em
cada conjunto - importante aqui pois "empate" tem só 16 exemplos.
Random state fixo (32) para reprodutibilidade.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

SEED_32 = 32
CAMINHO_ABORDAGEM2_32 = "data/features_abordagem2_32.csv"  # só para pegar id + classe
CAMINHO_SPLIT_32 = "data/split_32.csv"


def main_32():
    df_32 = pd.read_csv(CAMINHO_ABORDAGEM2_32)[["id_tabuleiro_32", "classe_32"]]

    treino_val_32, teste_32 = train_test_split(
        df_32, test_size=0.20, random_state=SEED_32, stratify=df_32["classe_32"]
    )
    # 0.25 de 80% = 20% do total -> fica 60/20/20
    treino_32, val_32 = train_test_split(
        treino_val_32, test_size=0.25, random_state=SEED_32,
        stratify=treino_val_32["classe_32"],
    )

    treino_32 = treino_32.copy()
    val_32 = val_32.copy()
    teste_32 = teste_32.copy()
    treino_32["conjunto_32"] = "treino"
    val_32["conjunto_32"] = "validacao"
    teste_32["conjunto_32"] = "teste"

    split_final_32 = pd.concat([treino_32, val_32, teste_32]).sort_values("id_tabuleiro_32")
    split_final_32[["id_tabuleiro_32", "conjunto_32"]].to_csv(CAMINHO_SPLIT_32, index=False)

    print("Divisão salva em", CAMINHO_SPLIT_32)
    print(split_final_32.groupby(["conjunto_32", "classe_32"]).size().unstack(fill_value=0))
    print("\nTotais por conjunto:")
    print(split_final_32["conjunto_32"].value_counts())


if __name__ == "__main__":
    main_32()
