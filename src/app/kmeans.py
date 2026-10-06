import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class KMeansClustering:
    
    def __init__(self, X_scaled, var_names=None, random_state=42):
        self.X = np.asarray(X_scaled)
        self.var_names = var_names
        self.random_state = random_state
        self.models = {}
        self.labels = {}
        self.results = []

    def evaluate_k(self, k_range=range(2, 6), n_init=50, max_iter=50):
        self.results = []
        for k in k_range:
            km = KMeans(
                n_clusters=k, init="k-means++", n_init=n_init,
                max_iter=max_iter, algorithm="lloyd",
                random_state=self.random_state,
            )
            km.fit(self.X)
            self.models[k] = km
            self.labels[k] = km.labels_
            sil = silhouette_score(self.X, km.labels_)
            self.results.append({
                "k": k, "inercia": km.inertia_, "silueta": sil,
            })
            print(f"k={k:>2}: inercia={km.inertia_:10.2f}  silueta={sil:.4f}")

        df = pd.DataFrame(self.results)
        best_k = df.loc[df["silueta"].idxmax(), "k"]
        print(f"\nMejor k por silueta: {int(best_k)}")
        return df

    def fit(self, k, n_init=50, max_iter=50):
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=n_init,
            max_iter=max_iter, algorithm="lloyd",
            random_state=self.random_state,
        )
        km.fit(self.X)
        self.models[k] = km
        self.labels[k] = km.labels_
        print(f"\n=== K-Means final (k={k}) ===")
        print(f"Inercia: {km.inertia_:.2f}")
        print(f"Silueta: {silhouette_score(self.X, km.labels_):.4f}")
        print("\nConteo por grupo:")
        print(pd.Series(km.labels_).value_counts().sort_index().to_string())
        return km

    def get_labels(self, k):
        return self.labels[k]

    def get_centroids(self, k, original_scale=True, scaler=None):
        km = self.models[k]
        centers = km.cluster_centers_
        if original_scale and scaler is not None:
            centers = scaler.inverse_transform(centers)
        df = pd.DataFrame(
            centers,
            columns=self.var_names if self.var_names else None,
        )
        print("\n=== Centroides ===")
        print(df.round(4).to_string())
        return df

    def plot_elbow(self, k_range=None):
        df = pd.DataFrame(self.results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(df["k"], df["inercia"], "o-", color="steelblue")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inercia")
        axes[0].set_title("Método del codo")
        axes[0].set_xticks(df["k"])

        axes[1].plot(df["k"], df["silueta"], "o-", color="darkorange")
        best_k = df.loc[df["silueta"].idxmax(), "k"]
        axes[1].axvline(x=best_k, color="red", linestyle="--", alpha=0.7,
                        label=f"Mejor k={int(best_k)}")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silueta")
        axes[1].set_title("Silueta vs k")
        axes[1].set_xticks(df["k"])
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_clusters_2d(self, k, title=None):
        labels = self.labels[k]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.X)
        if title is None:
            title = f"K-Means (k={k}) — ACP"
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels,
            cmap="Set1", alpha=0.7, edgecolors="white", s=60,
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
