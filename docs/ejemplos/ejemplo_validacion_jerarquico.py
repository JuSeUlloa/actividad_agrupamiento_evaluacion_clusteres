# %%
# Ejemplo Validación de Agrupamiento Jerárquico
# Determinación del número de grupos
# Conjunto de datos: HolzingerSwineford1939 {lavaan}
# Equivalente en Python del script R "Ejemplo Validacion Jerarquico HolzingerSwineford1939.R"

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt
import plotly.express as px
import pyreadr
import urllib.request
import tempfile
import os

# %%
# --- Dataset: HolzingerSwineford1939 ---

url = "https://raw.githubusercontent.com/yrosseel/lavaan/master/data/HolzingerSwineford1939.rda"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as response:
    rda_data = response.read()
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".rda")
tmp.write(rda_data)
tmp.close()
datos_raw = pyreadr.read_r(tmp.name)["HolzingerSwineford1939"]
os.unlink(tmp.name)

datos = datos_raw.dropna().copy()
datos = datos.rename(columns={
    "sex": "sexo", "school": "escuela", "grade": "grado",
    "x1": "visual", "x2": "cubos", "x3": "rombos",
    "x4": "parrafos", "x5": "oraciones", "x6": "palabras",
    "x7": "suma", "x8": "puntos", "x9": "letras",
})
datos["edad"] = datos["ageyr"] + datos["agemo"] / 12
datos["sexo"] = pd.Categorical(datos["sexo"].map({1: "Masculino", 2: "Femenino"}))
datos["escuela"] = pd.Categorical(datos["escuela"])
datos["grado"] = pd.Categorical(datos["grado"], categories=[7, 8], ordered=True)
datos = datos[["sexo", "edad", "escuela", "grado",
               "visual", "cubos", "rombos", "parrafos", "oraciones",
               "palabras", "suma", "puntos", "letras"]]

print("Dimensiones:", datos.shape)
print("\nPrimeras filas:")
print(datos.head(10))

# %%
# ======================================================================
# Funciones auxiliares
# ======================================================================

def gower_distance(df):
    """Distancia de Gower para variables mixtas (vectorizada)."""
    n = len(df)
    dist = np.zeros((n, n))
    for col in df.columns:
        vals = df[col].values
        if pd.api.types.is_numeric_dtype(df[col]):
            rng = np.nanmax(vals) - np.nanmin(vals)
            if rng == 0:
                continue
            dist += np.abs(vals[:, None] - vals[None, :]) / rng
        else:
            codes = pd.Categorical(vals).codes
            dist += (codes[:, None] != codes[None, :]).astype(float)
    dist /= df.shape[1]
    return dist

def dunn_index(X_scaled, labels, dist_matrix=None):
    """Índice de Dunn: mínima distancia inter-cluster / máxima distancia intra-cluster."""
    unique_labels = np.unique(labels)
    max_intra = 0
    for c in unique_labels:
        mask = labels == c
        if mask.sum() < 2:
            continue
        if dist_matrix is not None:
            sub = dist_matrix[np.ix_(mask, mask)]
            max_intra = max(max_intra, sub.max())
        else:
            sub = X_scaled[mask]
            from scipy.spatial.distance import pdist
            if len(sub) > 1:
                max_intra = max(max_intra, pdist(sub).max())

    min_inter = np.inf
    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i+1:]:
            mask1, mask2 = labels == c1, labels == c2
            if dist_matrix is not None:
                sub = dist_matrix[np.ix_(mask1, mask2)]
                min_inter = min(min_inter, sub.min())
            else:
                from scipy.spatial.distance import cdist
                d = cdist(X_scaled[mask1], X_scaled[mask2])
                min_inter = min(min_inter, d.min())

    return min_inter / max_intra if max_intra > 0 else 0

def gap_statistic(X_scaled, k_range, n_refs=20, random_state=42):
    """Estadística Gap: compara inercia real vs simulada."""
    rng = np.random.RandomState(random_state)
    gaps = []
    sks = []

    # Inercia real
    for k in k_range:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        inertia_real = km.inertia_

        # Inercia simulada
        inertia_sim = []
        data_min = X_scaled.min(axis=0)
        data_max = X_scaled.max(axis=0)
        for _ in range(n_refs):
            X_sim = rng.uniform(data_min, data_max, size=X_scaled.shape)
            km_sim = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_sim.fit(X_sim)
            inertia_sim.append(km_sim.inertia_)

        log_inertia_sim = np.log(np.array(inertia_sim))
        gap = np.mean(log_inertia_sim) - np.log(inertia_real)
        sk = np.sqrt(1 + 1 / len(inertia_sim)) * np.std(log_inertia_sim, ddof=1)
        gaps.append(gap)
        sks.append(sk)

    return np.array(gaps), np.array(sks)

def connectivity_score(X_scaled, labels, k_neighbors=10):
    """Conectividad: penaliza vecinos cercanos asignados a diferentes clusters."""
    from sklearn.neighbors import NearestNeighbors
    n = len(X_scaled)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_scaled)
    indices = nn.kneighbors(return_distance=False)

    conn = 0
    for i in range(n):
        for j in indices[i, 1:]:
            if labels[i] != labels[j]:
                conn += 1 / k_neighbors
    return conn

# %%
# ======================================================================
# PARTE 1: Variables numéricas
# ======================================================================
print("\n" + "=" * 60)
print("PARTE 1: Validación con variables numéricas")
print("=" * 60)

vars_num = ["visual", "cubos", "rombos", "parrafos", "oraciones",
            "palabras", "suma", "puntos", "letras"]
X_num = datos[vars_num].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# Matriz de distancias euclidean
from scipy.spatial.distance import pdist
D_euclidean = squareform(pdist(X_scaled, metric="euclidean"))

# Validación para k = 2:5
k_range = range(2, 6)
print("\n--- Métricas de validación (Ward + Euclidean) ---")
print(f"{'k':>4} {'Silueta':>10} {'Dunn':>10} {'CH':>10} {'DB':>10} {'Gap':>10} {'Conect.':>10}")

resultados = []
for k in k_range:
    Z = linkage(X_scaled, method="ward", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")

    sil = silhouette_score(X_scaled, labels)
    dunn = dunn_index(X_scaled, labels, D_euclidean)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    conn = connectivity_score(X_scaled, labels)

    resultados.append({
        "k": k, "silueta": sil, "dunn": dunn,
        "calinski_harabasz": ch, "davies_bouldin": db,
        "conectividad": conn,
    })
    print(f"{k:>4} {sil:>10.4f} {dunn:>10.4f} {ch:>10.2f} {db:>10.4f} {'---':>10} {conn:>10.2f}")

# Gap statistic
gaps, sks = gap_statistic(X_scaled, list(k_range))
print("\n--- Estadística Gap ---")
for i, k in enumerate(k_range):
    print(f"k={k}: Gap={gaps[i]:.4f}, S_k={sks[i]:.4f}")

print(f"\nMejor k por Gap: {k_range[np.argmax(gaps)]}")

# %%
def plot_silhouette_curve(X_scaled, k_range):
    """Gráfico de Silueta promedio vs k."""
    sil_scores = []
    for k in k_range:
        Z = linkage(X_scaled, method="ward", metric="euclidean")
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), sil_scores, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta promedio")
    ax.set_title("Silueta vs Número de Grupos")
    ax.set_xticks(list(k_range))
    best_k = list(k_range)[np.argmax(sil_scores)]
    ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.7,
               label=f"Mejor k={best_k}")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_silhouette_curve(X_scaled, k_range)

# %%
# Agrupamiento final con k seleccionado
k = 2
Z_final = linkage(X_scaled, method="ward", metric="euclidean")
grupos = fcluster(Z_final, t=k, criterion="maxclust")
datos["grupo_num"] = grupos

print(f"\n=== Agrupamiento final: k={k} ===")
print("\nConteo por grupo:")
print(pd.Series(grupos).value_counts().sort_index())

# %%
def plot_dendrogram_final(Z, k, title="Dendrograma"):
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=6,
               color_threshold=Z[-(k-1), 2], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Distancia")
    ax.axhline(y=Z[-(k-1), 2], color="red", linestyle="--", alpha=0.7,
               label=f"corte k={k}")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_dendrogram_final(Z_final, k, "Dendrograma - Numéricas (Ward, k=2)")

# %%
def plot_silhouette_bars(X_scaled, labels):
    """Gráfico de barras de silueta por individuo."""
    from sklearn.metrics import silhouette_samples
    sil_vals = silhouette_samples(X_scaled, labels)
    n_clusters = len(np.unique(labels))
    avg_sil = sil_vals.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 0
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    for i, c in enumerate(sorted(np.unique(labels))):
        cluster_sil = np.sort(sil_vals[labels == c])
        size = cluster_sil.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=avg_sil, color="red", linestyle="--",
               label=f"Silueta media = {avg_sil:.4f}")
    ax.set_xlabel("Coeficiente de Silueta")
    ax.set_ylabel("Cluster")
    ax.set_title("Silueta por Individuo")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_silhouette_bars(X_scaled, grupos)

# %%
def plot_clusters_2d(X_scaled, grupos, title="Clusters en ACP"):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=grupos,
                         cmap="Set1", alpha=0.7, edgecolors="white", s=60)
    for g in np.unique(grupos):
        mask = grupos == g
        cx, cy = coords[mask].mean(axis=0)
        ax.annotate(f"Grupo {g}", (cx, cy), fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Grupo")
    plt.tight_layout()
    plt.show()

def plot_clusters_3d(X_scaled, grupos, title="Clusters 3D"):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1], "PC3": coords[:, 2],
        "Grupo": pd.Categorical(grupos),
    })
    fig = px.scatter_3d(df_plot, x="PC1", y="PC2", z="PC3", color="Grupo",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        title=title)
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.show()

plot_clusters_2d(X_scaled, grupos, "Clusters en ACP - Numéricas (k=2)")
plot_clusters_3d(X_scaled, grupos, "Clusters 3D - Numéricas (k=2)")

# %%
# Descripción de los grupos
print("\n=== Descripción de los grupos (medias) ===")
desc_num = datos[vars_num].copy()
desc_num["grupo"] = grupos
print(desc_num.groupby("grupo").mean().round(2).to_string())

# %%
# ======================================================================
# PARTE 2: Variables mixtas (Gower)
# ======================================================================
print("\n" + "=" * 60)
print("PARTE 2: Validación con variables mixtas (Gower)")
print("=" * 60)

datos_mix = datos[["sexo", "edad", "escuela", "grado",
                    "visual", "cubos", "rombos", "parrafos", "oraciones",
                    "palabras", "suma", "puntos", "letras"]].copy()
datos_mix["grado"] = datos_mix["grado"].cat.codes

print("Calculando matriz de Gower...")
D_gower = gower_distance(datos_mix)
D_gower_condensed = squareform(D_gower)

k_range_gower = range(2, 6)
print("\n--- Métricas de validación (Ward + Gower) ---")
print(f"{'k':>4} {'Silueta':>10} {'Dunn':>10} {'CH':>10} {'DB':>10} {'Conect.':>10}")

for k in k_range_gower:
    Z_g = linkage(D_gower_condensed, method="ward")
    labels_g = fcluster(Z_g, t=k, criterion="maxclust")

    sil = silhouette_score(D_gower, labels_g, metric="precomputed")
    dunn = dunn_index(None, labels_g, D_gower)
    ch = calinski_harabasz_score(X_scaled, labels_g)  # CH usa euclidean
    db = davies_bouldin_score(X_scaled, labels_g)

    print(f"{k:>4} {sil:>10.4f} {dunn:>10.4f} {ch:>10.2f} {db:>10.4f}")

# %%
# Agrupamiento final con Gower (k=4 según el script R)
k_gower = 4
Z_gower = linkage(D_gower_condensed, method="ward")
grupos_gower = fcluster(Z_gower, t=k_gower, criterion="maxclust")
datos["grupo_gower"] = grupos_gower

print(f"\n=== Agrupamiento final Gower: k={k_gower} ===")
print("\nConteo por grupo:")
print(pd.Series(grupos_gower).value_counts().sort_index())

plot_dendrogram_final(Z_gower, k_gower, "Dendrograma - Mixtas (Gower + Ward, k=4)")

# %%
# Descripción de los grupos (Gower)
print("\n=== Descripción de los grupos (medias) ===")
desc_g = datos[["edad"] + vars_num].copy()
desc_g["grupo"] = grupos_gower
print(desc_g.groupby("grupo").mean().round(2).to_string())

print("\n--- Tablas de contingencia ---")
print("\nSexo × Grupo:")
print(pd.crosstab(datos["sexo"], grupos_gower))
print("\nEscuela × Grupo:")
print(pd.crosstab(datos["escuela"], grupos_gower))
print("\nGrado × Grupo:")
print(pd.crosstab(datos["grado"], grupos_gower))

print("\n=== Análisis completado ===")
