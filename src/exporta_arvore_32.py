"""
Exporta a Árvore de Decisão final (models/modelo_final_32.joblib) para JSON
puro (models/arvore_final_32.json).

Motivo: em alguns computadores (ex.: Windows com Controle de Aplicativo /
Smart App Control) o scipy, exigido pelo scikit-learn, é bloqueado e o
modelo .joblib não carrega. O JSON contém exatamente a mesma árvore
(mesmos nós, limiares e classes) e pode ser lido só com Python puro.
"""

import json

import joblib

CAMINHO_MODELO_32 = "models/modelo_final_32.joblib"
CAMINHO_JSON_32 = "models/arvore_final_32.json"


def main_32():
    pacote_32 = joblib.load(CAMINHO_MODELO_32)
    modelo_32 = pacote_32["modelo_32"]
    arvore_32 = modelo_32.tree_
    classes_32 = [str(c_32) for c_32 in modelo_32.classes_]

    nos_32 = []
    for no_32 in range(arvore_32.node_count):
        folha_32 = arvore_32.children_left[no_32] == -1
        classe_idx_32 = int(arvore_32.value[no_32][0].argmax())
        nos_32.append({
            "feature": None if folha_32 else int(arvore_32.feature[no_32]),
            "limiar": None if folha_32 else float(arvore_32.threshold[no_32]),
            "esquerda": None if folha_32 else int(arvore_32.children_left[no_32]),
            "direita": None if folha_32 else int(arvore_32.children_right[no_32]),
            "classe": classes_32[classe_idx_32],
        })

    with open(CAMINHO_JSON_32, "w", encoding="utf-8") as f_32:
        json.dump({"colunas": pacote_32["colunas_32"], "nos": nos_32}, f_32, ensure_ascii=False, indent=1)
    print(f"Árvore exportada: {len(nos_32)} nós -> {CAMINHO_JSON_32}")


if __name__ == "__main__":
    main_32()
