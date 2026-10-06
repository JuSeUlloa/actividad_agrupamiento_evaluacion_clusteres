# %%
# Ejemplo Medidas de Distancia
# Conjunto de datos: HolzingerSwineford1939 {lavaan}
# Equivalente en Python del script R "Ejemplo Distancias HolzingerSwineford1939.R"

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, jaccard
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
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

def gower_distance(df, types=None):
    """
    Distancia de Gower para variables mixtas.
    types: dict con 'factor', 'ordered', 'numeric', 'asymm', 'symm'
    """
    n = len(df)
    dist = np.zeros((n, n))
    n_vars = 0

    if types is None:
        types = {"numeric": list(df.columns)}

    # Variables numéricas
    if "numeric" in types:
        for col in types["numeric"]:
            vals = df[col].values.astype(float)
            rng = np.nanmax(vals) - np.nanmin(vals)
            if rng == 0:
                continue
            dist += np.abs(vals[:, None] - vals[None, :]) / rng
            n_vars += 1

    # Variables factor (categóricas nominales)
    if "factor" in types:
        for col in types["factor"]:
            codes = pd.Categorical(df[col]).codes
            dist += (codes[:, None] != codes[None, :]).astype(float)
            n_vars += 1

    # Variables ordered (ordinales)
    if "ordered" in types:
        for col in types["ordered"]:
            if pd.api.types.is_categorical_dtype(df[col]):
                codes = df[col].cat.codes.values.astype(float)
            else:
                codes = pd.Categorical(df[col]).codes.astype(float)
            max_code = codes.max()
            if max_code == 0:
                continue
            rng = max_code  # rango para ordinales
            dist += np.abs(codes[:, None] - codes[None, :]) / rng
            n_vars += 1

    # Variables binarias asimétricas (Jaccard)
    if "asymm" in types:
        for col in types["asymm"]:
            vals = df[col].values.astype(int)
            # Jaccard: dist = (b+c)/(a+b+c) donde a=ambos 1, b=0-1, c=1-0
            both_one = ((vals[:, None] == 1) & (vals[None, :] == 1)).astype(float)
            one_zero = ((vals[:, None] == 1) & (vals[None, :] == 0)).astype(float)
            zero_one = ((vals[:, None] == 0) & (vals[None, :] == 1)).astype(float)
            denom = both_one + one_zero + zero_one
            dist += np.where(denom > 0, (one_zero + zero_one) / denom, 0)
            n_vars += 1

    # Variables binarias simétricas (Afinidad)
    if "symm" in types:
        for col in types["symm"]:
            vals = df[col].values.astype(int)
            # Afinidad: dist = (b+c)/(a+b+c) + (d)/(a+b+c+d)
            both_one = ((vals[:, None] == 1) & (vals[None, :] == 1)).astype(float)
            both_zero = ((vals[:, None] == 0) & (vals[None, :] == 0)).astype(float)
            one_zero = ((vals[:, None] == 1) & (vals[None, :] == 0)).astype(float)
            zero_one = ((vals[:, None] == 0) & (vals[None, :] == 1)).astype(float)
            denom_sym = both_one + one_zero + zero_one + both_zero
            denom_asym = both_one + one_zero + zero_one
            dist += np.where(
                denom_asym > 0,
                (one_zero + zero_one) / denom_asym + both_zero / denom_sym,
                0,
            )
            n_vars += 1

    if n_vars > 0:
        dist /= n_vars
    return dist

def fviz_dist(D, title="Matriz de Distancias", figsize=(12, 10)):
    """Visualización de matriz de distancias (equivale a fviz_dist)."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        D, cmap="RdBu_r", center=0, square=True,
        linewidths=0, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# %%
# ======================================================================
# 1. Distancia para variables numéricas (Minkowski)
# ======================================================================
print("\n" + "=" * 60)
print("1. DISTANCIA PARA VARIABLES NUMÉRICAS")
print("=" * 60)

# Variables numéricas
vars_num = ["visual", "cubos", "rombos", "parrafos", "oraciones",
            "palabras", "suma", "puntos", "letras"]
X = datos[vars_num].copy()
print("\nPrimeras filas de X:")
print(X.head())

# Estandarizar
X_scaled = (X - X.mean()) / X.std()

# Distancia Euclidiana
D_euclidean = squareform(pdist(X_scaled.values, metric="euclidean"))
print("\n--- Distancia Euclidiana (primeras 5x5) ---")
print(pd.DataFrame(D_euclidean[:5, :5],
                   index=X.index[:5], columns=X.index[:5]).round(3))

# Distancia Manhattan
D_manhattan = squareform(pdist(X_scaled.values, metric="cityblock"))
print("\n--- Distancia Manhattan (primeras 5x5) ---")
print(pd.DataFrame(D_manhattan[:5, :5],
                   index=X.index[:5], columns=X.index[:5]).round(3))

fviz_dist(D_euclidean, "Distancia Euclidiana (estandarizada)")

# %%
# ======================================================================
# 2. Distancia para variables binarias
# ======================================================================
print("\n" + "=" * 60)
print("2. DISTANCIA PARA VARIABLES BINARIAS")
print("=" * 60)

# Binariar: puntaje <= 5 → 0, > 5 → 1
Y = (X > 5).astype(int)
# Quitar variables con un solo valor
Y = Y.loc[:, Y.nunique() > 1]
print("\nPrimeras filas de Y (binarizado):")
print(Y.head())

# Distancia de Jaccard (asimétrica)
# Solo importan los 1s compartidos
D_jaccard = squareform(pdist(Y.values, metric="jaccard"))
print("\n--- Distancia de Jaccard (primeras 5x5) ---")
print(pd.DataFrame(D_jaccard[:5, :5],
                   index=Y.index[:5], columns=Y.index[:5]).round(3))

# Distancia de Afinidad (simétrica)
# Importan tanto los 1s compartidos como los 0s compartidos
D_affinity = squareform(pdist(Y.values, metric="dice"))  # dice es similar a afinidad
print("\n--- Distancia de Afinidad/Dice (primeras 5x5) ---")
print(pd.DataFrame(D_affinity[:5, :5],
                   index=Y.index[:5], columns=Y.index[:5]).round(3))

fviz_dist(D_jaccard, "Distancia de Jaccard (binarias asimétricas)")

# %%
# ======================================================================
# 3. Distancia para variables categóricas (no ordinales)
# ======================================================================
print("\n" + "=" * 60)
print("3. DISTANCIA PARA VARIABLES CATEGÓRICAS (NO ORDINALES)")
print("=" * 60)

# Categorizar en bajo/medio/alto por cuantiles
Z = X.copy()
for col in Z.columns:
    quantiles = Z[col].quantile([0, 0.333, 0.666, 1]).values
    Z[col] = pd.cut(Z[col], bins=quantiles,
                    labels=["bajo", "medio", "alto"],
                    include_lowest=True)

print("\nPrimeras filas de Z (categorizado):")
print(Z.head())
print("\nDistribución de 'letras':")
print(Z["letras"].value_counts())

# Distancia de Gower con factor (nominal)
D_factor = gower_distance(Z, types={"factor": list(Z.columns)})
print("\n--- Distancia Gower-Factor (primeras 5x5) ---")
print(pd.DataFrame(D_factor[:5, :5],
                   index=Z.index[:5], columns=Z.index[:5]).round(3))

fviz_dist(D_factor, "Distancia Gower - Factor (nominal)")

# %%
# ======================================================================
# 4. Distancia para variables ordinales
# ======================================================================
print("\n" + "=" * 60)
print("4. DISTANCIA PARA VARIABLES ORDINALES")
print("=" * 60)

# Categorizar como ordinal
W = X.copy()
for col in W.columns:
    quantiles = W[col].quantile([0, 0.333, 0.666, 1]).values
    W[col] = pd.cut(W[col], bins=quantiles,
                    labels=["bajo", "medio", "alto"],
                    include_lowest=True, ordered=True)

print("\nPrimeras filas de W (ordinal):")
print(W.head())
print("\nDistribución de 'letras':")
print(W["letras"].value_counts().sort_index())

# Distancia de Gower con ordered
D_ordered = gower_distance(W, types={"ordered": list(W.columns)})
print("\n--- Distancia Gower-Ordered (primeras 5x5) ---")
print(pd.DataFrame(D_ordered[:5, :5],
                   index=W.index[:5], columns=W.index[:5]).round(3))

fviz_dist(D_ordered, "Distancia Gower - Ordered (ordinal)")

# %%
# ======================================================================
# 5. Distancia para mixtura de variables (Gower)
# ======================================================================
print("\n" + "=" * 60)
print("5. DISTANCIA PARA MIXTURA DE VARIABLES (GOWER)")
print("=" * 60)

# Preparar datos mixtos
datos_mix = datos[["sexo", "edad", "escuela", "grado"] + vars_num].copy()
# Convertir grado a código ordinal
datos_mix["grado"] = datos_mix["grado"].cat.codes

print("\nPrimeras filas de datos_mix:")
print(datos_mix.head())

# Gower con mixtura: factor(sexo, escuela), ordered(grado), numeric(edad, x1-x9)
D_gower = gower_distance(datos_mix, types={
    "factor": ["sexo", "escuela"],
    "ordered": ["grado"],
    "numeric": ["edad"] + vars_num,
})
print("\n--- Distancia Gower Mixta (primeras 5x5) ---")
print(pd.DataFrame(D_gower[:5, :5],
                   index=datos_mix.index[:5], columns=datos_mix.index[:5]).round(3))

fviz_dist(D_gower, "Distancia Gower - Mixtura de Variables")

print("\n=== Análisis completado ===")
