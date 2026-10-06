# %%
# Ejemplo K-Means y Algoritmos Similares
# Conjunto de datos: HolzingerSwineford1939 {lavaan}
# Equivalente en Python del script R "Ejemplo kmeans HolzingerSwineford1939.R"

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

# Limpiar y renombrar
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
# Variables numéricas
vars_num = ["visual", "cubos", "rombos", "parrafos", "oraciones",
            "palabras", "suma", "puntos", "letras"]
X_num = datos[vars_num].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# %%
# ======================================================================
# Funciones auxiliares de graficación
# ======================================================================

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

def plot_elbow(X_scaled, k_range=range(2, 11)):
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Método del Codo")
    axes[0].set_xticks(list(k_range))

    axes[1].plot(list(k_range), sil_scores, "o-", color="darkorange")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xticks(list(k_range))

    plt.tight_layout()
    plt.show()

# %%
# ======================================================================
# 1. K-Means (variables numéricas)
# ======================================================================
print("\n" + "=" * 60)
print("1. K-MEANS (variables numéricas)")
print("=" * 60)

k = 4
km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=30,
            algorithm="lloyd", random_state=42)
km.fit(X_scaled)
grupos_km = km.labels_

print(f"\nInercia: {km.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X_scaled, grupos_km):.4f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_km).value_counts().sort_index())

plot_clusters_2d(X_scaled, grupos_km, "K-Means (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_km, "K-Means (k=4) - 3D")
plot_elbow(X_scaled)

# %%
# ======================================================================
# 2. K-Means jerárquico (hk-means)
# ======================================================================
print("\n" + "=" * 60)
print("2. K-MEANS JERÁRQUICO (hk-means)")
print("=" * 60)

# Paso 1: clustering jerárquico para centroides iniciales
from scipy.cluster.hierarchy import linkage, fcluster

Z = linkage(X_scaled, method="ward", metric="euclidean")
centroides_init = np.zeros((k, X_scaled.shape[1]))
for g in range(k):
    mask = fcluster(Z, t=k, criterion="maxclust") == (g + 1)
    centroides_init[g] = X_scaled[mask].mean(axis=0)

# Paso 2: k-means con centroides iniciales del jerárquico
km_hk = KMeans(n_clusters=k, init=centroides_init, n_init=1, max_iter=30,
               algorithm="lloyd", random_state=42)
km_hk.fit(X_scaled)
grupos_hk = km_hk.labels_

print(f"\nInercia: {km_hk.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X_scaled, grupos_hk):.4f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_hk).value_counts().sort_index())

plot_clusters_2d(X_scaled, grupos_hk, "HK-Means (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_hk, "HK-Means (k=4) - 3D")

# %%
# ======================================================================
# 3. K-Medoides / PAM (variables numéricas)
# ======================================================================
print("\n" + "=" * 60)
print("3. K-MEDOIDES / PAM (variables numéricas)")
print("=" * 60)

from sklearn_extra.cluster import KMedoids

kmed = KMedoids(n_clusters=k, metric="euclidean", method="pam", random_state=42)
kmed.fit(X_scaled)
grupos_pam = kmed.labels_

print(f"\nInercia: {kmed.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X_scaled, grupos_pam):.4f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_pam).value_counts().sort_index())

plot_clusters_2d(X_scaled, grupos_pam, "PAM (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_pam, "PAM (k=4) - 3D")

# %%
# ======================================================================
# 4. Fuzzy C-Means (fanny)
# ======================================================================
print("\n" + "=" * 60)
print("4. FUZZY C-MEANS (fanny)")
print("=" * 60)

from skfuzzy import cmeans

# cmeans expects data in shape (n_features, n_samples)
cnorm, u, u0, d, jm, p, fpc = cmeans(X_scaled.T, c=k, m=1.1, error=0.005,
                                       maxiter=100, seed=42)
grupos_fuzzy = np.argmax(u, axis=0)
membership = u.max(axis=0)

print(f"\nFPC (Fuzzy Partition Coefficient): {fpc:.4f}")
print(f"Silhouette: {silhouette_score(X_scaled, grupos_fuzzy):.4f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_fuzzy).value_counts().sort_index())
print("\nGrados de pertenencia (primeros 5):")
print(pd.DataFrame(u.T, columns=[f"Grupo {i+1}" for i in range(k)]).head(5).round(4))

plot_clusters_2d(X_scaled, grupos_fuzzy, "Fuzzy C-Means (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_fuzzy, "Fuzzy C-Means (k=4) - 3D")

# %%
# ======================================================================
# 5. K-Prototipos (variables mixtas)
# ======================================================================
print("\n" + "=" * 60)
print("5. K-PROTOTIPOS (variables mixtas)")
print("=" * 60)

# kmodes/kprototypes para variables mixtas
from kmodes.kprototypes import KPrototypes

datos_mix = datos[["sexo", "edad", "escuela", "grado",
                    "visual", "cubos", "rombos", "parrafos", "oraciones",
                    "palabras", "suma", "puntos", "letras"]].copy()

# Preparar para kprototypes: columnas categóricas por índice
cat_cols_idx = [0, 2, 3]  # sexo, escuela, grado

kp = KPrototypes(n_clusters=k, init="Cao", n_init=5, verbose=0, random_state=42)
grupos_kp = kp.fit_predict(datos_mix.values, categorical=cat_cols_idx)

print(f"\nCosto: {kp.cost_:.2f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_kp).value_counts().sort_index())

# ACP para visualizar (solo numéricas)
plot_clusters_2d(X_scaled, grupos_kp, "K-Prototipos (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_kp, "K-Prototipos (k=4) - 3D")

# %%
# ======================================================================
# 6. K-Medoides con Gower (variables mixtas)
# ======================================================================
print("\n" + "=" * 60)
print("6. K-MEDOIDES con Gower (variables mixtas)")
print("=" * 60)

from scipy.spatial.distance import squareform

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

print("Calculando matriz de Gower...")
D_gower = gower_distance(datos_mix)
D_condensed = squareform(D_gower)

kmed_gower = KMedoids(n_clusters=k, metric="precomputed", method="pam", random_state=42)
kmed_gower.fit(D_gower)
grupos_kmed_gower = kmed_gower.labels_

print(f"\nInercia: {kmed_gower.inertia_:.2f}")
print("\nConteo por grupo:")
print(pd.Series(grupos_kmed_gower).value_counts().sort_index())

plot_clusters_2d(X_scaled, grupos_kmed_gower, "PAM Gower (k=4) - ACP")
plot_clusters_3d(X_scaled, grupos_kmed_gower, "PAM Gower (k=4) - 3D")

print("\n=== Análisis completado ===")
