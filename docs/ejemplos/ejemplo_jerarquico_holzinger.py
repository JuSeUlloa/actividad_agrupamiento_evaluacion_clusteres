# %%
# Ejemplo Agrupamiento Jerárquico
# Conjunto de datos: HolzingerSwineford1939 {lavaan}
# Equivalente en Python del script R "Ejemplo Jerarquico HolzingerSwineford1939.R"

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import pyreadr
import urllib.request
import tempfile
import os

# %%
# --- Dataset: HolzingerSwineford1939 ---
# 301 estudiantes de séptimo y octavo grado, pertenecientes a dos escuelas,
# en diferentes pruebas de habilidades mentales.
#
# Variables:
# - sex: sexo (1 = masculino, 2 = femenino)
# - ageyr: años cumplidos
# - agemo: meses adicionales
# - school: escuela (Pasteur o Grant-White)
# - grade: grado escolar (7 u 8)
# - x1-x9: pruebas de habilidades mentales

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
datos["grado"] = pd.Categorical(datos["grado"], categories=[7, 8],
                                 ordered=True)
datos = datos[["sexo", "edad", "escuela", "grado",
               "visual", "cubos", "rombos", "parrafos", "oraciones",
               "palabras", "suma", "puntos", "letras"]]

print("Dimensiones:", datos.shape)
print("\nPrimeras filas:")
print(datos.head(10))
print("\nTipos de variable:")
print(datos.dtypes)

# %%
# --- Funciones auxiliares ---

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
            diff = np.abs(vals[:, None] - vals[None, :]) / rng
            dist += diff
        else:
            codes = pd.Categorical(vals).codes
            diff = (codes[:, None] != codes[None, :]).astype(float)
            dist += diff

    dist /= df.shape[1]
    return dist

# %%
# ======================================================================
# PARTE 1: Agrupamiento jerárquico con variables numéricas
# ======================================================================

print("\n" + "=" * 60)
print("PARTE 1: Agrupamiento con variables numéricas")
print("=" * 60)

k = 4  # número de grupos

# Seleccionar variables numéricas y estandarizar
vars_num = ["visual", "cubos", "rombos", "parrafos", "oraciones",
            "palabras", "suma", "puntos", "letras"]
X_num = datos[vars_num].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# Distancia euclidiana + Ward
Z = linkage(X_scaled, method="ward", metric="euclidean")

# Asignación de grupos
grupos_num = fcluster(Z, t=k, criterion="maxclust")
datos["grupo_num"] = grupos_num

print(f"\nGrupo asignado (primeros 10): {grupos_num[:10]}")
print("\nConteo por grupo:")
print(pd.Series(grupos_num).value_counts().sort_index())

# %%
def plot_dendrogram(Z, labels=None, title="Dendrograma", filename="dendrograma.png"):
    """Dendrograma con partición en k grupos."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z, labels=labels, leaf_rotation=90, leaf_font_size=6,
        color_threshold=Z[-(k-1), 2], ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Distancia")
    ax.axhline(y=Z[-(k-1), 2], color="red", linestyle="--", alpha=0.7,
               label=f"corte k={k}")
    ax.legend()
    plt.tight_layout()
    
 #  plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

plot_dendrogram(Z, title="Dendrograma - Variables Numéricas (Ward)",
                filename="dendrograma_num.png")

# %%
def plot_clusters_pca(X_scaled, grupos, title="Clusters en ACP", save=True):
    """Representación de clusters en las primeras 2 componentes principales."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=grupos,
                         cmap="Set1", alpha=0.7, edgecolors="white", s=60)

    # Centroides
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
    #if save:
      #  
 #  plt.savefig("clusters_pca_num.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_clusters_pca(X_scaled, grupos_num, title="Clusters en ACP - Variables Numéricas")

# %%
def plot_clusters_3d(X_scaled, grupos, title="Clusters 3D (ACP)"):
    """Representación 3D de los clusters usando las primeras 3 componentes."""
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1], "PC3": coords[:, 2],
        "Grupo": pd.Categorical(grupos),
    })

    fig = px.scatter_3d(
        df_plot, x="PC1", y="PC2", z="PC3", color="Grupo",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=title,
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                "PC3": f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)"},
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.show()

plot_clusters_3d(X_scaled, grupos_num, title="Clusters 3D - Variables Numéricas")

# %%
# ======================================================================
# PARTE 2: Agrupamiento jerárquico con variables mixtas (Gower)
# ======================================================================

print("\n" + "=" * 60)
print("PARTE 2: Agrupamiento con variables mixtas (Gower)")
print("=" * 60)

k = 4

# Seleccionar variables y preparar tipos
# factor: sexo (col 0), escuela (col 2)
# ordered: grado (col 3)
# numeric: edad (col 1), visual:letras (cols 4-12)
datos_mix = datos[["sexo", "edad", "escuela", "grado",
                    "visual", "cubos", "rombos", "parrafos", "oraciones",
                    "palabras", "suma", "puntos", "letras"]].copy()

# Convertir ordered a codigos ordinales
datos_mix["grado"] = datos_mix["grado"].cat.codes

# Calcular matriz de Gower
print("\nCalculando matriz de distancia de Gower...")
dist_gower = gower_distance(datos_mix)
dist_condensed = squareform(dist_gower)

# Ward con Gower
Z_gower = linkage(dist_condensed, method="ward")

# Asignación de grupos
grupos_gower = fcluster(Z_gower, t=k, criterion="maxclust")
datos["grupo_gower"] = grupos_gower

print(f"\nGrupo asignado (primeros 10): {grupos_gower[:10]}")
print("\nConteo por grupo:")
print(pd.Series(grupos_gower).value_counts().sort_index())

# %%
plot_dendrogram(Z_gower, title="Dendrograma - Variables Mixtas (Gower + Ward)",
                filename="dendrograma_gower.png")

# %%
def plot_clusters_gower(X_scaled, grupos, title="Clusters en ACP (Gower)", save=True):
    """Representación de clusters Gower en ACP."""
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
    #if save:
      #  
 #  plt.savefig("clusters_pca_gower.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_clusters_gower(X_scaled, grupos_gower, title="Clusters en ACP - Variables Mixtas (Gower)")

# %%
plot_clusters_3d(X_scaled, grupos_gower, title="Clusters 3D - Variables Mixtas (Gower)")

print("\n=== Análisis completado ===")
print("Gráficos: dendrograma_num.png, clusters_pca_num.png,")
print("          dendrograma_gower.png, clusters_pca_gower.png")
