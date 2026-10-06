# AGRUPAMIENTO Y EVALUACIÓN DE CLÚSTERES

## Universidad Santo Tomás — Maestría en Ciencia de Datos

**Asignatura:** Métodos de Aprendizaje No Supervisado  
**Profesor:** Dr. MARIO JOSE PACHECO LOPEZ  
**Actividad 2:** Análisis de Conglomerados Aplicado a la Caracterización Morfométrica de Anuros  
**Dataset:** Leavey et al. (2023) — *Journal of Anatomy* ([DOI: 10.1111/joa.13886](https://doi.org/10.1111/joa.13886))

---

## Resumen ejecutivo

Este repositorio implementa en **Python** un plan de ciencia de datos no supervisado para la caracterización morfométrica y segmentación fenética de **N = 164 especies de ranas (anuros)**.

Se implementan y contrastan **tres familias de modelos no supervisados**:

1. **Agrupamiento Jerárquico Aglomerativo** — evaluación comparativa de enlaces Ward.D2, Complete, Average y Single.
2. **Agrupamiento Particional por K-Medias** — algoritmo Lloyd (equivalente a Hartigan-Wong en R) con múltiples reinicios.
3. **Partición Alrededor de Medoides (K-Medoides / PAM)** — con identificación de arquetipos biológicos observados reales.

La determinación del número óptimo de grupos ($k$) se fundamenta en baterías estadísticas completas de **validación interna** (Silueta, Dunn, Calinski-Harabasz, Davies-Bouldin, Conectividad) y **estabilidad estocástica** (Prediction Strength y remuestreo Bootstrap Jaccard), complementadas con **Gap Statistic** y **correlación cofenética**.

Asimismo, se evalúa formalmente la concordancia morfológica entre modelos y frente a la variable externa `habitat` mediante el **Índice de Rand Ajustado (ARI)** y tablas de contingencia ecológica con coeficiente Cramérv.

---

## Descripción del conjunto de datos

El archivo `joa13886-sup-0001-full dataset.xlsx` (hoja "Full dataset") recopila mediciones óseas tridimensionales obtenidas mediante microtomografía computarizada (μCT) en 164 especímenes de anuros adultos.

### Variables activas (18 dimensiones morfométricas)

| # | Variable | Descripción |
|---|----------|-------------|
| 1 | `skull` | Longitud anteroposterior del cráneo (cm) |
| 2 | `gap` | Espacio libre entre el cráneo y la columna vertebral (cm) |
| 3 | `vertebrae` | Longitud total de la columna presacra (cm) |
| 4 | `pelvis` | Longitud anteroposterior de la cintura pélvica (cm) |
| 5 | `ESD` | Expansión de las diapófisis sacras (cm) |
| 6 | `sacral_width` | Anchura total del sacro (cm) |
| 7 | `ilium` | Longitud del eje ilíaco (cm) |
| 8 | `urostyle` | Longitud del urostilo (cm) |
| 9 | `femur` | Longitud del fémur (cm) |
| 10 | `femur_width` | Diámetro diafisario del fémur (cm) |
| 11 | `tibiofibula` | Longitud del tibiofíbula (cm) |
| 12 | `calcaneus` | Longitud del calcáneo (cm) |
| 13 | `foot` | Longitud total del pie (cm) |
| 14 | `humerus` | Longitud del húmero (cm) |
| 15 | `humerus_width` | Diámetro diafisario del húmero (cm) |
| 16 | `radioulna` | Longitud del radio-cúbito (cm) |
| 17 | `hand` | Longitud total de la mano (cm) |
| 18 | `iliac_angle` | Ángulo de inserción ilíaca (°) |

### Variable de referencia externa

| Variable | Categorías | Distribución |
|----------|------------|--------------|
| `habitat` | Aquatic, Arboreal, Riparian, Terrestrial | 9, 28, 33, 94 |

> **Criterio metodológico:** `habitat` se mantiene **estrictamente separada** de la matriz de variables activas. No participa en la estandarización, ni en el cálculo de distancias, ni en la conformación de los clústeres.

---

## Decisiones metodológicas

| Decisión | Implementación | Justificación |
|----------|---------------|---------------|
| Variables activas | 18 variables morfométricas (skull → iliac_angle) | Alineado con el notebook R de referencia |
| Rango de k | `k = 2:5` | Rango de grupos biológicamente plausible y consistente con el estudio original |
| Enlace jerárquico principal | **Ward.D2** (`scipy` `method="ward"`) | Minimiza varianza intragrupal; produce grupos balanceados y mejor Davies-Bouldin |
| K-Means | **Lloyd** con `n_init=50`, `max_iter=50` | Equivalente disponible a Hartigan-Wong (no implementado en scikit-learn) |
| K-Medoides | **PAM** vía `sklearn_extra.cluster.KMedoids` | Homologación de `pam()` de R |
| Estandarización | Z-score (`StandardScaler`) | Homologación  `scale()` de R |
| Distancia | Euclidiana sobre datos estandarizados | Métrica natural para variables numéricas continuas |
| Métricas validación | Silueta, Dunn, CH, DB, Conectividad, PS, Jaccard + Gap + Cofenética | Las 7 del notebook más 2 adicionales como valor añadido |

> **Nota sobre Ward:** En R existen `ward.D` (variante antigua) y `ward.D2` (variante corregida). La implementación de scipy `method="ward"` equivale a `ward.D2` de R, que es la estándar recomendada.

---

## Objetivos

1. Preparar y describir la matriz de datos morfométricos (18 variables).
2. Estandarizar las variables mediante puntuaciones Z.
3. Calcular la matriz de distancias euclidiana.
4. Aplicar clustering jerárquico (Ward.D2, Complete, Average, Single) e interpretar dendrogramas.
5. Aplicar K-Means (Lloyd) y determinar el número óptimo de grupos.
6. Aplicar K-Medoides (PAM) e identificar los medoides representativos.
7. Validar los agrupamientos con Silueta, Dunn, Calinski-Harabasz, Davies-Bouldin, Conectividad, Prediction Strength, Bootstrap Jaccard, Gap Statistic y correlación cofenética.
8. Comparar las particiones (ARI) y analizar la asociación con `habitat` (Cramérv, chi-cuadrado).
9. Generar un informe reproducible que responda las preguntas planteadas.

---

## Metodología

### 1. Preparación de los datos — `DataPreparation`

- Carga el dataset Excel desde `data/raw/`.
- Selecciona las 18 variables morfométricas activas.
- Ejecuta chequeo de calidad: valores faltantes (NA), valores no finitos (Inf), varianza cero y cuasi-cero.
- Calcula estadísticos descriptivos (media, mediana, desviación estándar, mínimo, máximo, CV%).
- Estandariza las variables mediante puntuaciones Z.

### 2. Matriz de distancias — `DistanceMatrix`

- Calcula la distancia euclidiana sobre los datos estandarizados.
- Resume propiedades (media, mínima, máxima).
- Visualiza como heatmap reordenado.

### 3. Clustering jerárquico — `HierarchicalClustering`

- Construye dendrogramas con criterios de enlace **Ward.D2, Complete, Average y Single**.
- Calcula correlación cofenética para cada enlace.
- Permite cortar en k grupos y describir perfiles morfométricos.
- Genera proyección PCA 2D con áreas sombreadas (convex hull) por grupo.

### 4. K-Means — `KMeansClustering`

- Evalúa k ∈ [2, 5] con inercia y silueta.
- Ajusta el modelo final con **Lloyd** (`n_init=50`, `max_iter=50`).
- Calcula centroides en escala original y estandarizada.

### 5. K-Medoides (PAM) — `KMedoidsClustering`

- Aplica PAM con distancia euclidiana sobre la matriz estandarizada.
- Evalúa k ∈ [2, 5].
- Identifica índices de medoides y describe las observaciones representativas.

### 6. Validación — `ClusterValidation`

Implementa las siguientes métricas:

| Métrica | Código | Deseable |
|---------|--------|----------|
| Silueta promedio | `silueta` | Alto |
| Índice de Dunn | `dunn` | Alto |
| Calinski-Harabasz | `calinski_harabasz` | Máximo |
| Davies-Bouldin | `davies_bouldin` | Mínimo |
| Conectividad | `conectividad` | Mínimo |
| Prediction Strength | `prediction_strength` | Cercano a 1 |
| Bootstrap Jaccard | `bootstrap_jaccard` | Cercano a 1 |
| Gap Statistic | `gap` | Máximo |
| Correlación cofenética | `correlacion_cofenetica` | Cercano a 1 |

### 7. Comparación — `ClusterComparison`

- Calcula el **Adjusted Rand Index (ARI)** entre todas las particiones.
- Genera tablas de contingencia grupo × habitat.
- Calcula chi-cuadrado y coeficiente **Cramérv**.

---

## Estructura del repositorio

```text
actividad_agrupamiento_evaluacion_clusteres/
├── README.md                      # Este documento
├── main.py                        # Orquestador del análisis completo (Python)
├── pyproject.toml                 # Configuración del proyecto Python
├── .gitignore
├── .python-version
├── src/
│   ├── app/                       # Implementación en Python
│   │   ├── data_preparation.py    # DataPreparation
│   │   ├── distances.py           # DistanceMatrix
│   │   ├── hierarchical.py        # HierarchicalClustering
│   │   ├── kmeans.py              # KMeansClustering
│   │   ├── kmedoids.py            # KMedoidsClustering
│   │   ├── validation.py          # ClusterValidation
│   │   └── comparison.py          # ClusterComparison
│   └── R/                         # Implementación en R (notebook de referencia)
│       ├── Actividad_2_No_Supervisado.ipynb  # Notebook R completo
│       ├── vhc.R                  # Validación jerárquica
│       ├── vkm.R                  # Validación K-Means y K-Medoides
│       ├── install_packages.R     # Script de instalación de paquetes R
│       ├── environment_r.yml      # Entorno Conda para R
│       └── renv.lock              # Lockfile de renv
├── data/
│   ├── raw/                       # Dataset original (.xlsx)
│   └── processed/                 # Datos procesados
├── docs/
│   ├── ejemplos/                  # Scripts de referencia (Python)
│   ├── vhc.R                      # Función de validación jerárquica (R)
│   └── vkm.R                      # Función de validación particional (R)
└── figures/                       # Figuras generadas (PNG)
    ├── fig1_matriz_distancias.png
    ├── fig2_dendrograma.png
    ├── fig3_silueta.png
    └── fig4_clusters_pca.png
```

---

## Implementación en R (`src/R/`)

El repositorio incluye la implementación original en R que sirve como referencia para la versión en Python.

### Contenido de `src/R/`

| Archivo | Descripción |
|---------|-------------|
| `Actividad_2_No_Supervisado.ipynb` | Notebook R completo con el análisis completo |
| `vhc.R` | Función `vhc()` — validación de k para clustering jerárquico |
| `vkm.R` | Función `vkm()` — validación de k para K-Means y K-Medoides (PAM) |
| `install_packages.R` | Script para instalar paquetes R requeridos |
| `environment_r.yml` | Archivo de entorno Conda para R |
| `renv.lock` | Lockfile de renv para reproducibilidad |

### Funciones de validación en R

#### `vhc()` — Validación jerárquica

```r
vhc(D, k, m, met = "ward", X = NULL, B = 100, M = 100, nn = 10, dp = 1, seed = 123)
```

**Parámetros:**
- `D`: objeto `dist` o matriz de disimilitudes
- `k`: números de grupos a evaluar (ej: `2:5`)
- `m`: medidas a calcular (`"sil"`, `"dunn"`, `"ch"`, `"db"`, `"con"`, `"gap"`, `"ps"`, `"jac"`)
- `met`: método de enlace (`"ward"` se interpreta como `"ward.D2"`)
- `X`: matriz numérica (requerida para `db`, `gap`, `apn`, `fom`)
- `B`: remuestras para Gap y Jaccard
- `M`: particiones para Prediction Strength
- `nn`: vecinos para Connectivity
- `seed`: semilla

**Medidas disponibles:**
| Código | Medida |
|--------|--------|
| `sil` | Silhouette promedio |
| `dunn` | Índice de Dunn |
| `ch` | Calinski-Harabasz |
| `db` | Davies-Bouldin |
| `con` | Connectivity |
| `gap` | Gap statistic |
| `apn` | Average Proportion of Non-overlap |
| `fom` | Figure of Merit |
| `ps` | Prediction Strength |
| `jac` | Bootstrap Jaccard stability |

#### `vkm()` — Validación particional

```r
vkm(X = NULL, D = NULL, k, m, alg = "kmeans", B = 100, M = 100, nn = 10, ns = 25, seed = 123)
```

**Parámetros adicionales:**
- `alg`: `"kmeans"` o `"pam"`
- `ns`: número de inicios para k-means

### Paquetes R requeridos

| Paquete | Propósito |
|---------|-----------|
| `readxl` | Lectura de archivos Excel |
| `dplyr`, `tidyr` | Manipulación de datos |
| `ggplot2` | Gráficos estadísticos |
| `cluster` | Algoritmos de agrupamiento (daisy, pam) |
| `factoextra` | Visualización factorial |
| `FactoMineR` | PCA |
| `fpc` | Validación y clusterboot |
| `clValid` | Conectividad y estabilidad |
| `mclust` | ARI |
| `clusterSim`, `clusterCrit` | Davies-Bouldin |

### Instalación de dependencias R

```bash
cd src/R
Rscript install_packages.R
```

O usando Conda:

```bash
conda env create -f environment_r.yml
conda activate r_clustering
```

---

## Ejecución

### Requisitos previos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- Dataset en `data/raw/joa13886-sup-0001-full dataset.xlsx`

### Instalación de dependencias

```bash
cd actividad_agrupamiento_evaluacion_clusteres
uv sync
```

Las dependencias principales son:

- `numpy>=1.26.0,<2` (scikit-learn-extra requiere NumPy < 2)
- `scikit-learn>=1.9.0`
- `scikit-learn-extra>=0.3.0`
- `scipy>=1.17.0`
- `pandas>=3.0.0`
- `matplotlib`, `seaborn`
- `setuptools>=84.0.0` (provee `distutils` en Python 3.12)
- `openpyxl` (lectura de Excel)

### Ejecución del análisis completo

```bash
MPLBACKEND=Agg uv run python main.py
```

> **Nota:** `MPLBACKEND=Agg` permite ejecutar en entornos sin pantalla (servidores, CI). En un entorno local con display, se puede ejecutar simplemente `uv run python main.py`.

### Salida esperada

1. **Terminal:** Resultados numéricos de las 7 secciones del análisis.
2. **Figuras guardadas** en `figures/`:
   - `fig1_matriz_distancias.png` — Matriz de distancias euclídeas (heatmap)
   - `fig2_dendrograma.png` — Dendrograma Ward.D2 con corte en k=3
   - `fig3_silueta.png` — Coeficientes de silueta por individuo
   - `fig4_clusters_pca.png` — Proyección PCA con áreas sombreadas por grupo

### Secciones del análisis

| Sección | Contenido |
|---------|-----------|
| 1 | Preparación de datos, calidad, descriptivas, estandarización |
| 2 | Matriz de distancias euclídeas |
| 3 | Clustering jerárquico (4 enlaces), validación, dendrograma, silueta, PCA |
| 4 | K-Means (Lloyd): evaluación de k, centroides, perfiles |
| 5 | K-Medoides (PAM): evaluación de k, medoides, perfiles |
| 6 | Validación completa de K-Means y K-Medoides |
| 7 | Comparación (ARI), asociación con habitat (Cramérv, chi-cuadrado) |

---

## Resultados de referencia

Los resultados obtenidos con esta implementación reproducen los del notebook R de referencia:

| Método | k | Tamaños | Silueta | CH | DB |
|--------|---|---------|---------|-----|-----|
| Jerárquico (Ward.D2) | 3 | 79, 77, 8 | 0.399 | 142.380 | 0.862 |
| K-Means (Lloyd) | 3 | 66, 87, 11 | 0.407 | 151.351 | 0.896 |
| K-Medoides (PAM) | 3 | 64, 70, 30 | 0.336 | 134.249 | 1.012 |

**Concordancia entre métodos (ARI):**

| Comparación | ARI |
|-------------|-----|
| Jerárquico (Ward) vs. K-Means | 0.7873 |
| Jerárquico (Ward) vs. K-Medoides | 0.5127 |
| K-Means vs. K-Medoides | 0.4361 |

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `FileNotFoundError` al cargar datos | Verificar que el archivo esté en `data/raw/` con el nombre exacto `joa13886-sup-0001-full dataset.xlsx` |
| `ImportError: distutils` | Ejecutar `uv sync` para instalar `setuptools>=84.0.0` |
| `ConvergenceWarning` en KMedoids | Se incrementó `max_iter=500`; el warning es informativo |
| `FigureCanvasAgg is non-interactive` | Usar `MPLBACKEND=Agg` o ejecutar en entorno con display |
| Error de NumPy 2.x | Asegurar `numpy<2` en `pyproject.toml` (scikit-learn-extra no soporta NumPy 2) |

---

## Referencias

- Leavey, A., Ruta, M., Richards, C. T., & Porro, L. B. (2023). Locomotor, ecological and phylogenetic drivers of skeletal proportions in frogs. *Journal of Anatomy*, 243(3), 404–422. https://doi.org/10.1111/joa.13886
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- Hartigan, J. A., & Wong, M. A. (1979). Algorithm AS 136: A K-Means Clustering Algorithm. *Applied Statistics*, 28(1), 100–108.
- Tibshirani, R., & Walther, G. (2005). Cluster validation by prediction strength. *Journal of Computational and Graphical Statistics*, 14(3), 511–528.
- Documentación scikit-learn: https://scikit-learn.org/
- Documentación scipy: https://docs.scipy.org/

---
