import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class KMedoidsClustering:
   
    def __init__(self, X_scaled, var_names=None, random_state=42):
        self.X = np.asarray(X_scaled)
        self.var_names = var_names
        self.random_state = random_state
        self.models = {}    
        self.labels = {}    
        self.results = []

    def evaluate_k(self, k_range=range(2, 6), method="pam", max_iter=500):
        self.results = []
        for k in k_range:
            pam = KMedoids(
                n_clusters=k, metric="euclidean", method=method,
                max_iter=max_iter, random_state=self.random_state,
            )
            pam.fit(self.X)
            self.models[k] = pam
            self.labels[k] = pam.labels_
            sil = silhouette_score(self.X, pam.labels_)
            self.results.append({
                "k": k, "inercia": pam.inertia_, "silueta": sil,
            })
            print(f"k={k:>2}: inercia={pam.inertia_:10.2f}  silueta={sil:.4f}")

        df = pd.DataFrame(self.results)
        best_k = df.loc[df["silueta"].idxmax(), "k"]
        print(f"\nMejor k por silueta: {int(best_k)}")
        return df

    def fit(self, k, method="pam", max_iter=500):
        pam = KMedoids(
            n_clusters=k, metric="euclidean", method=method,
            max_iter=max_iter, random_state=self.random_state,
        )
        pam.fit(self.X)
        self.models[k] = pam
        self.labels[k] = pam.labels_
        print(f"\n=== K-Medoides / PAM final (k={k}) ===")
        print(f"Inercia: {pam.inertia_:.2f}")
        print(f"Silueta: {silhouette_score(self.X, pam.labels_):.4f}")
        print("\nÍndices de medoides:", pam.medoid_indices_)
        print("\nConteo por grupo:")
        print(pd.Series(pam.labels_).value_counts().sort_index().to_string())
        return pam

    def get_labels(self, k):
        return self.labels[k]

    def get_medoids(self, k, data_original=None, var_names=None):
        pam = self.models[k]
        idx = pam.medoid_indices_
        print("\n=== Medoides (índices y mediciones originales) ===")
        if data_original is not None and var_names is not None:
            df = data_original.iloc[idx][var_names].copy()
            df.insert(0, "indice_original", idx)
            df.insert(1, "grupo", pam.labels_[idx])
            print(df.round(4).to_string())
            return df
        else:
            print("Índices:", idx)
            return idx

    def plot_elbow(self):
        df = pd.DataFrame(self.results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(df["k"], df["inercia"], "o-", color="steelblue")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inercia")
        axes[0].set_title("K-Medoides — Inercia")
        axes[0].set_xticks(df["k"])

        axes[1].plot(df["k"], df["silueta"], "o-", color="darkorange")
        best_k = df.loc[df["silueta"].idxmax(), "k"]
        axes[1].axvline(x=best_k, color="red", linestyle="--", alpha=0.7,
                        label=f"Mejor k={int(best_k)}")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silueta")
        axes[1].set_title("K-Medoides — Silueta vs k")
        axes[1].set_xticks(df["k"])
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_clusters_2d(self, k, title=None):
        labels = self.labels[k]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.X)
        if title is None:
            title = f"K-Medoides (k={k}) — ACP"
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels,
            cmap="Set1", alpha=0.7, edgecolors="white", s=60,
        )
        pam = self.models[k]
        medoid_coords = coords[pam.medoid_indices_]
        ax.scatter(
            medoid_coords[:, 0], medoid_coords[:, 1],
            c="black", marker="X", s=200, edgecolors="yellow",
            linewidths=2, zorder=5, label="Medoides",
        )
        for g in np.unique(labels):
            mask = labels == g
            cx, cy = coords[mask].mean(axis=0)
            ax.annotate(
                f"Grupo {g}", (cx, cy), fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(title)
        ax.legend(*scatter.legend_elements(), title="Grupo")
        plt.tight_layout()
        plt.show()

    def group_profiles(self, k, data_original, var_names):
        df = data_original[var_names].copy()
        df["grupo"] = self.labels[k]
        profiles = df.groupby("grupo").mean().round(3)
        print("\n=== Perfiles de grupos (medias originales) ===")
        print(profiles.to_string())
        return profiles
