import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns


class DistanceMatrix:
    

    def __init__(self, X_scaled, var_names=None):
        self.X = np.asarray(X_scaled)
        self.var_names = var_names
        self.D = None
        self.D_condensed = None

    def compute(self, metric="euclidean"):

        self.D_condensed = pdist(self.X, metric=metric)
        self.D = squareform(self.D_condensed)
        print(f"\n=== Matriz de distancias ({metric}) ===")
        print(f"Forma: {self.D.shape}")
        print(f"Distancia media: {self.D_condensed.mean():.4f}")
        print(f"Distancia mínima: {self.D_condensed.min():.4f}")
        print(f"Distancia máxima: {self.D_condensed.max():.4f}")
        return self.D

    def get_condensed(self):
        return self.D_condensed

    def get_square(self):
        return self.D

    def summary_table(self, n=5):
        
        if self.D is None:
            raise RuntimeError("Primero llama a compute()")
        idx = range(min(n, self.D.shape[0]))
        df = pd.DataFrame(
            self.D[np.ix_(list(idx), list(idx))],
            index=[f"obs{i+1}" for i in idx],
            columns=[f"obs{i+1}" for i in idx],
        )
        print(f"\n--- Distancias (primeras {n}×{n}) ---")
        print(df.round(3).to_string())
        return df

    def plot(self, title="Matriz de distancias", save_path=None):
        if self.D is None:
            raise RuntimeError("Primero llama a compute()")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            self.D, cmap="RdBu_r", center=0, square=True,
            linewidths=0, ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figura guardada: {save_path}")
        plt.show()
