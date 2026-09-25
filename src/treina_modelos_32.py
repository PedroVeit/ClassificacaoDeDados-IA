"""
Etapa 5 do trabalho: treina e avalia 5 algoritmos classificadores nas
2 abordagens de pré-processamento, usando o MESMO split (treino /
validação / teste) para todos.

Algoritmos (5, conforme exigido):
  1. k-NN (obrigatório)
  2. Árvore de Decisão (obrigatório)
  3. MLP (obrigatório) - topologia reportada no log/print
  4. Random Forest (livre escolha 1) - ensemble de árvores de decisão
     treinadas em subamostras (bagging) e subconjuntos de features;
     a predição final é o voto majoritário das árvores. Reduz overfitting
     em relação a uma única árvore.
  5. SVM (livre escolha 2) - Support Vector Machine: busca o hiperplano
     (ou fronteira não-linear, via kernel RBF) que maximiza a margem
     entre as classes no espaço de features. Para múltiplas classes,
     o scikit-learn usa a estratégia "um-contra-um" internamente.

Para cada algoritmo x abordagem, testamos uma pequena grade de
hiperparâmetros no conjunto de TREINO, escolhemos a melhor combinação
pela acurácia no conjunto de VALIDAÇÃO (evitando overfitting), e só
então avaliamos a combinação vencedora no conjunto de TESTE (que nunca
é usado para decidir parâmetros).

Métricas reportadas no teste: acurácia, precision, recall e F1
(médias "macro", que tratam todas as classes com o mesmo peso -
importante aqui pois 'empate' tem poucos exemplos).
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED_32 = 32
CAMINHO_SPLIT_32 = "data/split_32.csv"
ABORDAGENS_32 = {
    "abordagem1": "data/features_abordagem1_32.csv",
    "abordagem2": "data/features_abordagem2_32.csv",
}
PRECISA_ESCALA_32 = {"knn", "mlp", "svm"}

GRADES_32 = {
    "knn": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "arvore_decisao": {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
    "mlp": {
        "hidden_layer_sizes": [(8,), (16,), (16, 8), (32, 16)],
        "learning_rate_init": [0.01, 0.05],
    },
    "random_forest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "svm": {"C": [0.5, 1, 5, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
}


def constroi_modelo_32(nome_32, params_32):
    if nome_32 == "knn":
        return KNeighborsClassifier(**params_32)
    if nome_32 == "arvore_decisao":
        return DecisionTreeClassifier(random_state=SEED_32, **params_32)
    if nome_32 == "mlp":
        return MLPClassifier(
            random_state=SEED_32, max_iter=3000, early_stopping=False, **params_32
        )
    if nome_32 == "random_forest":
        return RandomForestClassifier(random_state=SEED_32, **params_32)
    if nome_32 == "svm":
        return SVC(random_state=SEED_32, **params_32)
    raise ValueError(nome_32)


def carrega_abordagem_32(caminho_features_32):
    feats_32 = pd.read_csv(caminho_features_32)
    split_32 = pd.read_csv(CAMINHO_SPLIT_32)
    df_32 = feats_32.merge(split_32, on="id_tabuleiro_32")
    colunas_x_32 = [c_32 for c_32 in feats_32.columns if c_32 not in ("id_tabuleiro_32", "classe_32")]

    treino_32 = df_32[df_32["conjunto_32"] == "treino"]
    val_32 = df_32[df_32["conjunto_32"] == "validacao"]
    teste_32 = df_32[df_32["conjunto_32"] == "teste"]

    return (
        treino_32[colunas_x_32].values, treino_32["classe_32"].values,
        val_32[colunas_x_32].values, val_32["classe_32"].values,
        teste_32[colunas_x_32].values, teste_32["classe_32"].values,
        colunas_x_32,
    )


def escala_se_necessario_32(nome_32, X_treino_32, X_val_32, X_teste_32):
    if nome_32 not in PRECISA_ESCALA_32:
        return X_treino_32, X_val_32, X_teste_32
    escalador_32 = StandardScaler().fit(X_treino_32)
    return (
        escalador_32.transform(X_treino_32),
        escalador_32.transform(X_val_32),
        escalador_32.transform(X_teste_32),
    )


def escolhe_melhores_parametros_32(nome_32, X_treino_32, y_treino_32, X_val_32, y_val_32):
    melhor_acc_32 = -1
    melhor_params_32 = None
    for params_32 in ParameterGrid(GRADES_32[nome_32]):
        modelo_32 = constroi_modelo_32(nome_32, params_32)
        modelo_32.fit(X_treino_32, y_treino_32)
        acc_val_32 = accuracy_score(y_val_32, modelo_32.predict(X_val_32))
        if acc_val_32 > melhor_acc_32:
            melhor_acc_32 = acc_val_32
            melhor_params_32 = params_32
    return melhor_params_32, melhor_acc_32


def main_32():
    resultados_32 = []
    matrizes_confusao_32 = {}

    for nome_abordagem_32, caminho_32 in ABORDAGENS_32.items():
        (X_treino_32, y_treino_32, X_val_32, y_val_32,
         X_teste_32, y_teste_32, colunas_32) = carrega_abordagem_32(caminho_32)

        for nome_algoritmo_32 in GRADES_32:
            Xt_32, Xv_32, Xte_32 = escala_se_necessario_32(
                nome_algoritmo_32, X_treino_32, X_val_32, X_teste_32
            )

            inicio_32 = time.time()
            melhores_params_32, acc_val_32 = escolhe_melhores_parametros_32(
                nome_algoritmo_32, Xt_32, y_treino_32, Xv_32, y_val_32
            )

            # treina o modelo final com treino+validação usando os melhores parâmetros,
            # e avalia no teste (nunca usado para escolher parâmetros)
            X_treino_val_32 = np.vstack([Xt_32, Xv_32])
            y_treino_val_32 = np.concatenate([y_treino_32, y_val_32])
            modelo_final_32 = constroi_modelo_32(nome_algoritmo_32, melhores_params_32)
            modelo_final_32.fit(X_treino_val_32, y_treino_val_32)
            y_pred_teste_32 = modelo_final_32.predict(Xte_32)
            tempo_total_32 = time.time() - inicio_32

            resultados_32.append({
                "abordagem_32": nome_abordagem_32,
                "algoritmo_32": nome_algoritmo_32,
                "melhores_params_32": json.dumps(melhores_params_32, default=str),
                "acuracia_validacao_32": round(acc_val_32, 4),
                "acuracia_teste_32": round(accuracy_score(y_teste_32, y_pred_teste_32), 4),
                "precision_teste_32": round(
                    precision_score(y_teste_32, y_pred_teste_32, average="macro", zero_division=0), 4
                ),
                "recall_teste_32": round(
                    recall_score(y_teste_32, y_pred_teste_32, average="macro", zero_division=0), 4
                ),
                "f1_teste_32": round(
                    f1_score(y_teste_32, y_pred_teste_32, average="macro", zero_division=0), 4
                ),
                "tempo_treino_segundos_32": round(tempo_total_32, 3),
                "n_features_32": len(colunas_32),
            })
            matrizes_confusao_32[f"{nome_abordagem_32}__{nome_algoritmo_32}"] = confusion_matrix(
                y_teste_32, y_pred_teste_32,
                labels=sorted(set(y_teste_32)),
            ).tolist()
            print(f"[{nome_abordagem_32:12s}] {nome_algoritmo_32:16s} "
                  f"val_acc={acc_val_32:.3f}  teste_acc={resultados_32[-1]['acuracia_teste_32']:.3f}  "
                  f"params={melhores_params_32}")

    df_resultados_32 = pd.DataFrame(resultados_32)
    df_resultados_32.to_csv("reports/resultados_modelos_32.csv", index=False)

    with open("reports/matrizes_confusao_32.json", "w") as f_32:
        json.dump({
            "labels_ordem_32": sorted(set(y_teste_32)),
            "matrizes_32": matrizes_confusao_32,
        }, f_32, indent=2)

    print("\n=== TABELA RESUMO (ordenada por acurácia no teste) ===")
    print(df_resultados_32.sort_values("acuracia_teste_32", ascending=False)
          [["abordagem_32", "algoritmo_32", "acuracia_teste_32", "f1_teste_32", "tempo_treino_segundos_32"]]
          .to_string(index=False))

    melhor_linha_32 = df_resultados_32.sort_values(
        ["acuracia_teste_32", "f1_teste_32"], ascending=False
    ).iloc[0]
    print(f"\n>>> Melhor combinação: {melhor_linha_32['algoritmo_32']} "
          f"na {melhor_linha_32['abordagem_32']} "
          f"(acurácia teste = {melhor_linha_32['acuracia_teste_32']})")


if __name__ == "__main__":
    main_32()
