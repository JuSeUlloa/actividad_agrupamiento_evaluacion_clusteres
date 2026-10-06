# %%
# Ejemplo Validación de Agrupamiento K-Means
# Determinación del número de grupos
# Conjunto de datos: HolzingerSwineford1939 {lavaan}
# Equivalente en Python del script R "Ejemplo Validacion kmeans HolzingerSwineford1939.R"

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, silhouette_samples,
)
from sklearn.neighbors import NearestNeighbors
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
# Funciones auxiliares de validación
# ======================================================================

def dunn_index(X_scaled=None, labels=None, dist_matrix=None):
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
            from scipy.spatial.distance import pdist
            sub = X_scaled[mask]
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
    gaps, sks = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        inertia_real = km.inertia_

        data_min, data_max = X_scaled.min(axis=0), X_scaled.max(axis=0)
        inertia_sim = []
        for _ in range(n_refs):
            X_sim = rng.uniform(data_min, data_max, size=X_scaled.shape)
            km_sim = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_sim.fit(X_sim)
            inertia_sim.append(km_sim.inertia_)

        log_sim = np.log(np.array(inertia_sim))
        gap = np.mean(log_sim) - np.log(inertia_real)
        sk = np.sqrt(1 + 1 / len(inertia_sim)) * np.std(log_sim, ddof=1)
        gaps.append(gap)
        sks.append(sk)
    return np.array(gaps), np.array(sks)

def connectivity_score(X_scaled, labels, k_neighbors=10):
    """Conectividad: penaliza vecinos cercanos en diferentes clusters."""
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_scaled)
    indices = nn.kneighbors(return_distance=False)
    conn = 0
    for i in range(len(X_scaled)):
        for j in indices[i, 1:]:
            if labels[i] != labels[j]:
                conn += 1 / k_neighbors
    return conn

def gower_distance(df, types=None):
    """Distancia de Gower para variables mixtas."""
    n = len(df)
    dist = np.zeros((n, n))
    n_vars = 0
    if types is None:
        types = {"numeric": list(df.columns)}

    if "numeric" in types:
        for col in types["numeric"]:
            vals = df[col].values.astype(float)
            rng = np.nanmax(vals) - np.nanmin(vals)
            if rng == 0:
                continue
            dist += np.abs(vals[:, None] - vals[None, :]) / rng
            n_vars += 1

    if "factor" in types:
        for col in types["factor"]:
            codes = pd.Categorical(df[col]).codes
            dist += (codes[:, None] != codes[None, :]).astype(float)
            n_vars += 1

    if "ordered" in types:
        for col in types["ordered"]:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                codes = df[col].cat.codes.values.astype(float)
            else:
                codes = pd.Categorical(df[col]).codes.astype(float)
            max_code = codes.max()
            if max_code == 0:
                continue
            dist += np.abs(codes[:, None] - codes[None, :]) / max_code
            n_vars += 1

    if n_vars > 0:
        dist /= n_vars
    return dist

# %%
# ======================================================================
# Funciones auxiliares de graficación (sin save)
# ======================================================================

def plot_silhouette_curve(X_scaled, k_range, algorithm="kmeans"):
    sil_scores = []
    for k in k_range:
        if algorithm == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            labels = km.labels_
        else:
            Z = linkage(X_scaled, method="ward", metric="euclidean")
            from scipy.cluster.hierarchy import fcluster
            labels = fcluster(Z, t=k, criterion="maxclust")
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), sil_scores, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta promedio")
    ax.set_title(f"Silueta vs k ({algorithm})")
    ax.set_xticks(list(k_range))
    best_k = list(k_range)[np.argmax(sil_scores)]
    ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.7,
               label=f"Mejor k={best_k}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_silhouette_bars(X_scaled, labels):
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

# %%
# ======================================================================
# PARTE 1: K-Means con variables numéricas
# ======================================================================
print("\n" + "=" * 60)
print("PARTE 1: Validación K-Means (variables numéricas)")
print("=" * 60)

vars_num = ["visual", "cubos", "rombos", "parrafos", "oraciones",
            "palabras", "suma", "puntos", "letras"]
X_num = datos[vars_num].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# Validación para k = 2:5
k_range = range(2, 6)
print("\n--- Métricas de validación (K-Means) ---")
print(f"{'k':>4} {'Silueta':>10} {'Dunn':>10} {'CH':>10} {'DB':>10} {'Gap':>10} {'Conect.':>10}")

gaps, sks = gap_statistic(X_scaled, list(k_range))

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, max_iter=30, random_state=42)
    km.fit(X_scaled)
    labels = km.labels_

    sil = silhouette_score(X_scaled, labels)
    dunn = dunn_index(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    conn = connectivity_score(X_scaled, labels)
    gap_val = gaps[list(k_range).index(k)]

    print(f"{k:>4} {sil:>10.4f} {dunn:>10.4f} {ch:>10.2f} {db:>10.4f} {gap_val:>10.4f} {conn:>10.2f}")

print("\n--- Estadística Gap ---")
for i, k in enumerate(k_range):
    print(f"k={k}: Gap={gaps[i]:.4f}, S_k={sks[i]:.4f}")

plot_silhouette_curve(X_scaled, k_range, algorithm="kmeans")

# %%
# Agrupamiento final con k=2
k = 2
km_final = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=30,
                  algorithm="lloyd", random_state=42)
km_final.fit(X_scaled)
grupos_km = km_final.labels_
datos["grupo_kmeans"] = grupos_km

print(f"\n=== Agrupamiento final K-Means: k={k} ===")
print(f"Inercia: {km_final.inertia_:.2f}")
print(f"Silueta: {silhouette_score(X_scaled, grupos_km):.4f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_km).value_counts().sort_index())

# Silueta por individuo
plot_silhouette_bars(X_scaled, grupos_km)

# Representación en ACP
plot_clusters_2d(X_scaled, grupos_km, "K-Means (k=2) - ACP")
plot_clusters_3d(X_scaled, grupos_km, "K-Means (k=2) - 3D")

# Descripción de los grupos
print("\n=== Descripción de los grupos (medias) ===")
desc_km = datos[vars_num].copy()
desc_km["grupo"] = grupos_km
print(desc_km.groupby("grupo").mean().round(2).to_string())

# %%
# ======================================================================
# PARTE 2: K-Medoides con variables mixtas (Gower)
# ======================================================================
print("\n" + "=" * 60)
print("PARTE 2: Validación K-Medoides con Gower (variables mixtas)")
print("=" * 60)

from sklearn_extra.cluster import KMedoids

datos_mix = datos[["sexo", "edad", "escuela", "grado"] + vars_num].copy()
datos_mix["grado"] = datos_mix["grado"].cat.codes

print("\nCalculando matriz de Gower...")
D_gower = gower_distance(datos_mix, types={
    "factor": ["sexo", "escuela"],
    "ordered": ["grado"],
    "numeric": ["edad"] + vars_num,
})

# Validación para k = 2:5 con PAM
print("\n--- Métricas de validación (PAM + Gower) ---")
print(f"{'k':>4} {'Silueta':>10} {'Dunn':>10} {'CH':>10} {'DB':>10}")

for k in k_range:
    pam = KMedoids(n_clusters=k, metric="precomputed", method="pam", random_state=42)
    pam.fit(D_gower)
    labels = pam.labels_

    sil = silhouette_score(D_gower, labels, metric="precomputed")
    dunn = dunn_index(labels=labels, dist_matrix=D_gower)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f"{k:>4} {sil:>10.4f} {dunn:>10.4f} {ch:>10.2f} {db:>10.4f}")

# Agrupamiento final con k=2
k = 2
pam_final = KMedoids(n_clusters=k, metric="precomputed", method="pam", random_state=42)
pam_final.fit(D_gower)
grupos_pam = pam_final.labels_
datos["grupo_pam"] = grupos_pam

print(f"\n=== Agrupamiento final PAM+Gower: k={k} ===")
print(f"Inercia: {pam_final.inertia_:.2f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_pam).value_counts().sort_index())

# Descripción de los grupos
print("\n=== Descripción de los grupos (medias) ===")
desc_pam = datos[["edad"] + vars_num].copy()
desc_pam["grupo"] = grupos_pam
print(desc_pam.groupby("grupo").mean().round(2).to_string())

print("\n--- Tablas de contingencia ---")
print("\nSexo × Grupo:")
print(pd.crosstab(datos["sexo"], grupos_pam))
print("\nEscuela × Grupo:")
print(pd.crosstab(datos["escuela"], grupos_pam))
print("\nGrado × Grupo:")
print(pd.crosstab(datos["grado"], grupos_pam))

# Representación en ACP
plot_clusters_2d(X_scaled, grupos_pam, "PAM+Gower (k=2) - ACP")
plot_clusters_3d(X_scaled, grupos_pam, "PAM+Gower (k=2) - 3D")

print("\n=== Análisis completado ===")
