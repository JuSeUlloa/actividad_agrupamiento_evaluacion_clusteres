

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.app.data_preparation import DataPreparation
from src.app.distances import DistanceMatrix
from src.app.hierarchical import HierarchicalClustering
from src.app.kmeans import KMeansClustering
from src.app.kmedoids import KMedoidsClustering
from src.app.validation import ClusterValidation
from src.app.comparison import ClusterComparison

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)

K_RANGE = range(2, 6)  # Alineado con el notebook R (k = 2:5)
K_OPT = 3
# El notebook usa ward.D2 como enlace principal: produce grupos balanceados
# y con mejor separación (DB), aunque "average" tenga silueta más alta
# (esta última genera efecto cadena con grupos degenerados).
HIER_METHOD = "ward"

# Directorio de figuras
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    # ================================================================
    # 1. PREPARACIÓN DE DATOS
    # ================================================================
    print("=" * 60)
    print("SECCIÓN 1: PREPARACIÓN DE DATOS")
    print("=" * 60)

    data_prep = DataPreparation()
    data_prep.load_data()
    data_prep.quality_check()
    data_prep.describe()
    X_scaled = data_prep.standardize()

    # ================================================================
    # 2. DISTANCIAS
    # ================================================================
    print("\n" + "=" * 60)
    print("SECCIÓN 2: MATRIZ DE DISTANCIAS")
    print("=" * 60)

    dist = DistanceMatrix(X_scaled.values, var_names=data_prep.var_names)
    D = dist.compute(metric="euclidean")
    dist.summary_table(n=5)
    # Figura 1: Matriz de distancias (como fviz_dist del notebook)
    dist.plot(
        title="Figura 1. Matriz de Distancias Euclídeas Estandarizadas (164 especies de Anuros)",
        save_path=FIGURES_DIR / "fig1_matriz_distancias.png",
    )

    # ================================================================
    # 3. CLUSTERING JERÁRQUICO
    # ================================================================

    print("\n" + "=" * 60)
    print("SECCIÓN 3: CLUSTERING JERÁRQUICO")
    print("=" * 60)

    print("\n--- Clustering jerárquico aglomerativo ---")

    hier = HierarchicalClustering(X_scaled.values, var_names=data_prep.var_names)
    hier.fit(methods=("single", "complete", "average", "ward"))

    cophenetic_corr = hier.cophenetic_all()

    val_hier = ClusterValidation(X_scaled.values, dist_matrix=D)
    hier_results = {}

    for method in ("single", "complete", "average", "ward"):
        df_m = val_hier.evaluate_range(
            K_RANGE, method="hierarchical", linkage_method=method,
        )
        df_m = val_hier.add_gap(df_m, list(K_RANGE))
        df_m["correlacion_cofenetica"] = cophenetic_corr[method]
        hier_results[method] = df_m

    best_method = HIER_METHOD
    print(f"\nEnlace principal seleccionado: {best_method}")
    print("(Ward minimiza varianza intragrupal; produce grupos balanceados)")

    labels_ward = hier.cut(K_OPT, method=best_method)
    hier.group_profiles(labels_ward, data_prep.data, data_prep.var_names)

    # Figura 2: Dendrograma ward (como fviz_dend del notebook)
    hier.plot_dendrogram(
        method=best_method, k=K_OPT,
        title="Figura 2. Dendrograma Jerárquico Aglomerativo (Ward.D2, k = 3)",
        save_path=FIGURES_DIR / "fig2_dendrograma.png",
    )

    # Figura 3: Silueta individual (como fviz_silhouette del notebook)
    val_hier.plot_silhouette_bars(
        labels_ward,
        title="Figura 3. Coeficientes de Silueta por Individuo (Jerárquico, k = 3)",
        save_path=FIGURES_DIR / "fig3_silueta.png",
    )

    # Figura 4: Proyección PCA (como fviz_cluster del notebook)
    hier.plot_clusters_2d(
        labels_ward,
        title="Figura 4. Proyección Factorial 2D de los Grupos Jerárquicos sobre ACP",
        save_path=FIGURES_DIR / "fig4_clusters_pca.png",
    )

    # ================================================================
    # 4. K-MEANS
    # ================================================================
    print("\n" + "=" * 60)
    print("SECCIÓN 4: K-MEANS")
    print("=" * 60)

    km = KMeansClustering(X_scaled.values, var_names=data_prep.var_names)
    km_results = km.evaluate_k(k_range=K_RANGE)

    km.fit(K_OPT)
    km.get_centroids(K_OPT, original_scale=True, scaler=data_prep.scaler)
    km.group_profiles(K_OPT, data_prep.data, data_prep.var_names)
    labels_km = km.get_labels(K_OPT)

    # ================================================================
    # 5. K-MEDOIDES (PAM)
    # ================================================================
    print("\n" + "=" * 60)
    print("SECCIÓN 5: K-MEDOIDES / PAM")
    print("=" * 60)

    pam = KMedoidsClustering(X_scaled.values, var_names=data_prep.var_names)
    pam_results = pam.evaluate_k(k_range=K_RANGE)

    pam.fit(K_OPT)
    pam.get_medoids(K_OPT, data_original=data_prep.data, var_names=data_prep.var_names)
    pam.group_profiles(K_OPT, data_prep.data, data_prep.var_names)
    labels_pam = pam.get_labels(K_OPT)

    # ================================================================
    # 6. VALIDACIÓN K-MEANS Y K-MEDOIDES
    # ================================================================
    print("\n" + "=" * 60)
    print("SECCIÓN 6: VALIDACIÓN K-MEANS Y K-MEDOIDES")
    print("=" * 60)

    val = ClusterValidation(X_scaled.values, dist_matrix=D)

    df_km = val.evaluate_range(K_RANGE, method="kmeans")
    df_km = val.add_gap(df_km, list(K_RANGE))

    df_pam = val.evaluate_range(K_RANGE, method="pam")
    df_pam = val.add_gap(df_pam, list(K_RANGE))

    # ================================================================
    # 7. COMPARACIÓN Y HABITAT
    # ================================================================
    print("\n" + "=" * 60)
    print("SECCIÓN 7: COMPARACIÓN Y ASOCIACIÓN CON HABITAT")
    print("=" * 60)

    comp = ClusterComparison()
    comp.add_partition(f"Jerárquico-{best_method}", labels_ward)
    comp.add_partition("K-Means", labels_km)
    comp.add_partition("K-Medoides", labels_pam)

    comp.ari_matrix()

    df_hier_best = hier_results[best_method]
    siluetas = {
        f"Jerárquico-{best_method}": float(df_hier_best.loc[df_hier_best["k"] == K_OPT, "silueta"].iloc[0]),
        "K-Means": float(df_km.loc[df_km["k"] == K_OPT, "silueta"].iloc[0]),
        "K-Medoides": float(df_pam.loc[df_pam["k"] == K_OPT, "silueta"].iloc[0]),
    }
    comp.summary_table(siluetas=siluetas)

    if data_prep.habitat is not None:
        for name in comp.partitions:
            tabla, chi2, p, cramers_v = comp.contingency(name, data_prep.habitat)

    print("\n=== Análisis completado ===")


if __name__ == "__main__":
    main()
