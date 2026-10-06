import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class HierarchicalClustering:


    def __init__(self, X_scaled, var_names=None, random_state=42):
        self.X = np.asarray(X_scaled)
        self.var_names = var_names
        self.random_state = random_state
        self.linkages = {}
        self.labels = {}
        self.k_opt = None
        self.D_condensed = pdist(self.X, metric="euclidean")

    def fit(self, methods=("single", "complete", "average", "ward"), metric="euclidean"):

        for m in methods:
            Z = linkage(self.X, method=m, metric=metric)
            self.linkages[m] = Z
            print(f"Enlace '{m}': altura máxima = {Z[-1, 2]:.4f}")
        return self

    def cophenetic_correlation(self, method="ward"):
        Z = self.linkages[method]
        result = cophenet(Z, self.D_condensed)
        coph_corr = float(result[0])
        print(f"\nCorrelación cofenética ({method}): {coph_corr:.4f}")
        return coph_corr

    def cophenetic_all(self):
        results = {}
        print("\n=== Correlación cofenética por enlace ===")
        for method in self.linkages:
            results[method] = self.cophenetic_correlation(method)
        return results

    def get_linkage(self, method="ward"):
        return self.linkages[method]

    def cut(self, k, method="ward"):
        
        Z = self.linkages[method]
        labels = fcluster(Z, t=k, criterion="maxclust")
        self.labels[(method, k)] = labels
        print(f"\n--- Corte k={k} ({method}) ---")
        print(pd.Series(labels).value_counts().sort_index().to_string())
        return labels

    def plot_dendrogram(self, method="ward", k=None, title=None, figsize=(14, 6), save_path=None):
        Z = self.linkages[method]
        if title is None:
            title = f"Dendrograma — enlace {method}"
        fig, ax = plt.subplots(figsize=figsize)
        color_threshold = Z[-(k - 1), 2] if k and k > 1 else None
        dendrogram(
            Z, leaf_rotation=90, leaf_font_size=6,
            color_threshold=color_threshold, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Observaciones")
        ax.set_ylabel("Distancia")
        if k and k > 1:
            ax.axhline(y=Z[-(k - 1), 2], color="red", linestyle="--", alpha=0.7,
                       label=f"corte k={k}")
            ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figura guardada: {save_path}")
        plt.show()

    def plot_clusters_2d(self, labels, title="Clusters en ACP", method="ward", k=None, save_path=None):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.X)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Colores por grupo
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        color_map = {g: colors[i] for i, g in enumerate(unique_labels)}

        # Dibujar áreas sombreadas (convex hull) por grupo
        for g in unique_labels:
            mask = labels == g
            points = coords[mask]
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = Polygon(
                    hull_points,
                    closed=True,
                    facecolor=color_map[g],
                    alpha=0.2,
                    edgecolor=color_map[g],
                    linewidth=2,
                    linestyle="--",
                    label=f"Grupo {g}" if g == unique_labels[0] else None,
                )
                ax.add_patch(polygon)

        # Scatter de observaciones
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels,
            cmap="Set1", alpha=0.7, edgecolors="white", s=60,
        )

        # Etiquetas de grupo en el centroide
        for g in unique_labels:
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

        # Leyenda: mostrar patches de convex hull
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map[g], alpha=0.3, edgecolor=color_map[g],
                  linestyle="--", label=f"Grupo {g}")
            for g in unique_labels
        ]
        ax.legend(handles=legend_elements, title="Grupo", loc="best")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figura guardada: {save_path}")
        plt.show()

    def group_profiles(self, labels, data_original, var_names):
        df = data_original[var_names].copy()
        df["grupo"] = labels
        profiles = df.groupby("grupo").mean().round(3)
        print("\n=== Perfiles de grupos (medias originales) ===")
        print(profiles.to_string())
        return profiles
