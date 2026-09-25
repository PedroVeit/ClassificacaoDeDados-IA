"""
Treina o modelo FINAL escolhido para o front end e salva em disco.

Escolha: Árvore de Decisão na abordagem2 (7 features de alto nível).
Justificativa (para o relatório):
  - Empatou em 1º lugar em acurácia de teste (96,77%) com kNN, MLP,
    Random Forest e SVM na abordagem2 (ver reports/resultados_modelos_32.csv).
  - É a mais rápida para treinar e, principalmente, para PREVER a cada
    jogada em tempo real no front end (não precisa calcular distâncias
    como o kNN nem fazer forward-pass como o MLP).
  - É interpretável (dá para inspecionar as regras aprendidas), o que
    ajuda a explicar o comportamento da IA na apresentação/vídeo.
  - Usa a abordagem2, que é "menos custosa" (7 features) e generalizou
    muito melhor que a abordagem1 (9 features brutas) com a mesma
    quantidade de dados de treino.

Este modelo final é treinado com TODOS os tabuleiros do dataset_final_32
(treino+validação+teste), pois os números de desempenho já foram
reportados corretamente (com conjunto de teste isolado) em
treina_modelos_32.py / reports/resultados_modelos_32.csv. Usar mais
dados aqui só melhora o modelo que efetivamente roda no front end.
"""

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

SEED_32 = 32
MELHORES_PARAMS_32 = {"max_depth": 8, "min_samples_leaf": 1}
CAMINHO_MODELO_32 = "models/modelo_final_32.joblib"


def main_32():
    df_32 = pd.read_csv("data/features_abordagem2_32.csv")
    colunas_x_32 = [c_32 for c_32 in df_32.columns if c_32 not in ("id_tabuleiro_32", "classe_32")]

    modelo_32 = DecisionTreeClassifier(random_state=SEED_32, **MELHORES_PARAMS_32)
    modelo_32.fit(df_32[colunas_x_32], df_32["classe_32"])

    joblib.dump({"modelo_32": modelo_32, "colunas_32": colunas_x_32}, CAMINHO_MODELO_32)
    print(f"Modelo final salvo em {CAMINHO_MODELO_32}")
    print(f"Features esperadas (ordem): {colunas_x_32}")


if __name__ == "__main__":
    main_32()
