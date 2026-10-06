import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreparation:
    """
    Carga, inspección y preparación del dataset de ranas
    (Leavey et al., 2023 — Journal of Anatomy).
    """

    # Rutas relativas a la raíz del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATASET_FILENAME = "joa13886-sup-0001-full dataset.xlsx"

    # Variables morfométricas activas (18 dimensiones, skull → iliac_angle)
    VARS_MORFOMETRICAS = [
        "skull", "gap", "vertebrae", "pelvis", "ESD", "sacral_width", "ilium",
        "urostyle", "femur", "femur_width", "tibiofibula", "calcaneus", "foot",
        "humerus", "humerus_width", "radioulna", "hand", "iliac_angle"
    ]
    VAR_HABITAT = "habitat"
    HABITAT_LEVELS = ["Aquatic", "Arboreal", "Riparian", "Terrestrial"]

    def __init__(self):
        self.data = None
        self.data_scaled = None
        self.scaler = StandardScaler()
        self.var_names = None
        self.habitat = None

    def get_numeric_variables(self):
        return self.data[self.var_names]

    def get_habitat(self):
        return self.habitat

    def get_scaled_data(self):
        return self.data_scaled

    def get_matrix(self):
        return self.data_scaled.values if isinstance(self.data_scaled, pd.DataFrame) else self.data_scaled

    def load_data(self, path=None):
        if path is None:
            path = self.DATA_RAW / self.DATASET_FILENAME
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset: {path}\n Descárgalo y colócalo en {self.DATA_RAW}/"
            )

        self.data = pd.read_excel(path)

        # Preservar nombres originales de columnas (ESD tiene mayúsculas)
        self.data.columns = [c.strip().replace(" ", "_") for c in self.data.columns]

        # Mapeo explícito por si el Excel variara en nombres
        col_map = {c.lower(): c for c in self.data.columns}
        disponibles = []
        for v in self.VARS_MORFOMETRICAS:
            if v in self.data.columns:
                disponibles.append(v)
            elif v.lower() in col_map:
                disponibles.append(col_map[v.lower()])
        if not disponibles:
            num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.var_names = num_cols
        else:
            self.var_names = disponibles

        if self.VAR_HABITAT in self.data.columns:
            self.habitat = self.data[self.VAR_HABITAT]
        else:
            hab_cols = [c for c in self.data.columns if "habitat" in c.lower()]
            if hab_cols:
                self.habitat = self.data[hab_cols[0]]
            else:
                self.habitat = None

        # Ordenar niveles de habitat (misma convención que el notebook R)
        if self.habitat is not None:
            self.habitat = pd.Categorical(
                self.habitat, categories=self.HABITAT_LEVELS, ordered=True
            )

        self.data = self.data.dropna(subset=self.var_names).reset_index(drop=True)
        if self.habitat is not None:
            self.habitat = self.habitat[self.data.index]

        print(f"Dimensiones del dataset: {self.data.shape}")
        print(f"Variables activas ({len(self.var_names)}): {self.var_names}")
        if self.habitat is not None:
            print(f"\nDistribución de {self.VAR_HABITAT}:")
            print(self.habitat.value_counts().to_string())
        return self.data

    def quality_check(self):
        """Diagnóstico de calidad de datos (alineado con el notebook R)."""
        print("\n=== Chequeo de calidad de datos ===")
        sub = self.data[self.var_names]

        n_na = int(sub.isna().sum().sum())
        n_inf = int(np.isinf(sub.to_numpy()).sum())
        var_cero = [v for v in self.var_names if sub[v].var() == 0]
        var_cuasi = [v for v in self.var_names if 0 < sub[v].var() < 1e-4]

        print(f"Valores faltantes (NA): {n_na}")
        print(f"Valores no finitos (Inf): {n_inf}")
        print(f"Variables con varianza exactamente cero: {len(var_cero)} {var_cero if var_cero else ''}")
        print(f"Variables con varianza cuasi-cero (Var < 1e-4): {len(var_cuasi)} {var_cuasi if var_cuasi else ''}")
        return {
            "n_na": n_na,
            "n_inf": n_inf,
            "var_cero": var_cero,
            "var_cuasi_cero": var_cuasi,
        }

    def describe(self):
        desc = self.data[self.var_names].describe().loc[["mean", "50%", "std", "min", "max"]]
        desc = desc.rename(index={"50%": "median"})
        # Coeficiente de variación porcentual (como en el notebook R)
        desc.loc["cv_pct"] = (desc.loc["std"] / desc.loc["mean"]) * 100
        print("\n=== Estadísticas descriptivas ===")
        print(desc.round(3).to_string())
        return desc

    def correlation_matrix(self):
        corr = self.data[self.var_names].corr()
        print("\n=== Matriz de correlaciones ===")
        print(corr.round(3).to_string())

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Matriz de correlaciones (variables morfométricas)")
        plt.tight_layout()
        plt.show()
        return corr

    def standardize(self):
        X = self.data[self.var_names].values
        self.data_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=self.var_names
        )
        print("\n=== Datos estandarizados ===")
        print(self.data_scaled.describe().round(4).to_string())
        return self.data_scaled
