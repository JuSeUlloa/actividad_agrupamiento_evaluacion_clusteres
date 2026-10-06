import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


class ClusterComparison:

    def __init__(self):
        self.partitions = {}  

    def add_partition(self, name, labels):
        self.partitions[name] = np.asarray(labels)

    def ari_matrix(self):
        names = list(self.partitions.keys())
        n = len(names)
        mat = np.zeros((n, n))
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                mat[i, j] = adjusted_rand_score(
                    self.partitions[n1], self.partitions[n2]
                )
        df = pd.DataFrame(mat, index=names, columns=names)
        print("\n=== Adjusted Rand Index (ARI) entre particiones ===")
        print(df.round(4).to_string())
        return df

    def contingency(self, name, external, external_name="habitat"):
        labels = self.partitions[name]
        tabla = pd.crosstab(labels, external, rownames=["grupo"],
                            colnames=[external_name])
        print(f"\n=== Contingencia: {name} × {external_name} ===")
        print(tabla.to_string())

        chi2, p, dof, _ = chi2_contingency(tabla)
        n = tabla.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1))) if min(tabla.shape) > 1 else 0
        print(f"\nChi-cuadrado = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
        print(f"Cramérv = {cramers_v:.4f}")
        return tabla, chi2, p, cramers_v

    def plot_contingency(self, name, external, external_name="habitat"):
        labels = self.partitions[name]
        tabla = pd.crosstab(labels, external)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} × {external_name}")
        ax.set_ylabel("Grupo")
        ax.set_xlabel(external_name)
        plt.tight_layout()
        plt.show()

    def summary_table(self, siluetas=None):
        names = list(self.partitions.keys())
        tamanos = []
        for n in names:
            labs = self.partitions[n]
            tamanos.append(len(np.unique(labs)))
        df = pd.DataFrame({
            "metodo": names,
            "k": tamanos,
        })
        if siluetas:
            df["silueta"] = df["metodo"].map(siluetas)
        print("\n=== Resumen de métodos ===")
        print(df.to_string(index=False))
        return df
