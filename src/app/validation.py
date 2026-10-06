import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, silhouette_samples,
)
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class ClusterValidation:

    def __init__(self, X_scaled, dist_matrix=None):
        self.X = np.asarray(X_scaled)
        self.D = dist_matrix

    # ------------------------------------------------------------------
    # Generación de etiquetas según método
    # ------------------------------------------------------------------
    def _get_labels(self, method, k, linkage_method="ward", random_state=42):
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            model.fit(self.X)
            return model.labels_
        elif method == "hierarchical":
            Z = linkage(self.X, method=linkage_method, metric="euclidean")
            return fcluster(Z, t=k, criterion="maxclust")
        elif method == "pam":
            from sklearn_extra.cluster import KMedoids
            model = KMedoids(n_clusters=k, metric="euclidean", method="pam",
                             random_state=random_state)
            model.fit(self.X)
            return model.labels_
        else:
            raise ValueError(f"Método no soportado: {method}")

    # ------------------------------------------------------------------
    # Índices individuales
    # ------------------------------------------------------------------
    def silhouette(self, labels):
        if self.D is not None:
            return silhouette_score(self.D, labels, metric="precomputed")
        return silhouette_score(self.X, labels)

    def dunn(self, labels):
        unique = np.unique(labels)
        max_intra = 0.0
        for c in unique:
            mask = labels == c
            if mask.sum() < 2:
                continue
            if self.D is not None:
                sub = self.D[np.ix_(mask, mask)]
                max_intra = max(max_intra, sub.max())
            else:
                sub = self.X[mask]
                max_intra = max(max_intra, cdist(sub, sub).max())

        min_inter = np.inf
        for i, c1 in enumerate(unique):
            for c2 in unique[i + 1:]:
                m1, m2 = labels == c1, labels == c2
                if self.D is not None:
                    sub = self.D[np.ix_(m1, m2)]
                    min_inter = min(min_inter, sub.min())
                else:
                    min_inter = min(min_inter, cdist(self.X[m1], self.X[m2]).min())

        return min_inter / max_intra if max_intra > 0 else 0.0

    def calinski_harabasz(self, labels):
        return calinski_harabasz_score(self.X, labels)

    def davies_bouldin(self, labels):
        return davies_bouldin_score(self.X, labels)

    def connectivity(self, labels, k_neighbors=10):
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(self.X)
        indices = nn.kneighbors(return_distance=False)
        conn = 0.0
        for i in range(len(self.X)):
            for j in indices[i, 1:]:
                if labels[i] != labels[j]:
                    conn += 1.0 / k_neighbors
        return conn

    def gap_statistic(self, k_range, n_refs=20, random_state=42):
        rng = np.random.RandomState(random_state)
        gaps, sks = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km.fit(self.X)
            inertia_real = km.inertia_

            data_min, data_max = self.X.min(axis=0), self.X.max(axis=0)
            inertia_sim = []
            for _ in range(n_refs):
                X_sim = rng.uniform(data_min, data_max, size=self.X.shape)
                km_sim = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                km_sim.fit(X_sim)
                inertia_sim.append(km_sim.inertia_)

            log_sim = np.log(np.array(inertia_sim))
            gap = np.mean(log_sim) - np.log(inertia_real)
            sk = np.sqrt(1 + 1 / len(inertia_sim)) * np.std(log_sim, ddof=1)
            gaps.append(gap)
            sks.append(sk)
        return np.array(gaps), np.array(sks)

    def prediction_strength(self, k, method="kmeans", linkage_method="ward",
                            train_frac=0.5, random_state=42):
        """
        Prediction Strength (Tibshirani & Walther, 2005).
        Divide la muestra en entrenamiento y validación, ajusta el clustering
        en cada mitad y verifica cuántos pares permanecen juntos en validación.
        Deseable: valor cercano a 1.
        """
        rng = np.random.RandomState(random_state)
        n = len(self.X)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx, val_idx = idx[:split], idx[split:]

        X_train, X_val = self.X[train_idx], self.X[val_idx]

        # Clustering en entrenamiento
        labels_train = self._fit_labels(
            X_train, k, method, linkage_method, random_state
        )

        # Clustering en validación
        labels_val = self._fit_labels(
            X_val, k, method, linkage_method, random_state
        )

        # Centroides del entrenamiento (o medoides reales)
        centroids = np.array([
            X_train[labels_train == c].mean(axis=0)
            for c in range(k)
        ]) if method != "hierarchical" else None

        if method == "hierarchical":
            Z_val = linkage(X_val, method=linkage_method, metric="euclidean")
            labels_val_pred = fcluster(Z_val, t=k, criterion="maxclust")
        else:
            from scipy.spatial.distance import cdist as _cdist
            dists = _cdist(X_val, centroids)
            labels_val_pred = np.argmin(dists, axis=1)

        
        labels_val_indep = self._fit_labels(
            X_val, k, method, linkage_method, random_state
        )

        
        ps_values = []
        for c in range(1, k + 1):
            mask_c = labels_val_indep == c
            n_c = mask_c.sum()
            if n_c < 2:
                continue
            pairs_same_pred = 0
            total_pairs = 0
            members = np.where(mask_c)[0]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    total_pairs += 1
                    if labels_val_pred[members[a]] == labels_val_pred[members[b]]:
                        pairs_same_pred += 1
            if total_pairs > 0:
                ps_values.append(pairs_same_pred / total_pairs)

        return float(np.min(ps_values)) if ps_values else 0.0

    def _fit_labels(self, X, k, method, linkage_method, random_state):
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            model.fit(X)
            return model.labels_
        elif method == "hierarchical":
            Z = linkage(X, method=linkage_method, metric="euclidean")
            return fcluster(Z, t=k, criterion="maxclust")
        elif method == "pam":
            from sklearn_extra.cluster import KMedoids
            model = KMedoids(n_clusters=k, metric="euclidean", method="pam",
                             random_state=random_state)
            model.fit(X)
            return model.labels_
        else:
            raise ValueError(f"Método no soportado: {method}")

    def bootstrap_jaccard(self, labels, n_boot=50, method="kmeans",
                          linkage_method="ward", random_state=42):
        rng = np.random.RandomState(random_state)
        n = len(self.X)
        k = len(np.unique(labels))
        n_boot = min(n_boot, n)

        jaccard_matrix = np.full((n_boot, k), np.nan)

        for b in range(n_boot):
            boot_idx = rng.choice(n, size=n, replace=True)
            X_boot = self.X[boot_idx]
            labels_boot = self._fit_labels(X_boot, k, method, linkage_method,
                                           random_state)

            for c_orig in range(1, k + 1):
                mask_orig = labels == c_orig
                members_orig = set(np.where(mask_orig)[0])

                best_jaccard = 0.0
                for c_boot in np.unique(labels_boot):
                    mask_boot = labels_boot == c_boot
                    boot_members = set(boot_idx[mask_boot]) & members_orig
                    union = len(members_orig | set(boot_idx[mask_boot]))
                    if union > 0:
                        j = len(boot_members) / union
                        best_jaccard = max(best_jaccard, j)
                jaccard_matrix[b, c_orig - 1] = best_jaccard

        jaccard_mean = np.nanmean(jaccard_matrix, axis=0)
        jaccard_min = np.nanmin(jaccard_matrix, axis=0)
        jaccard_global = float(np.nanmean(jaccard_matrix))

        result = {
            "jaccard_global": jaccard_global,
            "jaccard_mean_by_group": jaccard_mean,
            "jaccard_min_by_group": jaccard_min,
        }
        return result

    def evaluate_range(self, k_range, method="kmeans", linkage_method="ward",
                       labels_func=None, compute_stability=True,
                       n_boot=30, random_state=42):
        results = []
        for k in k_range:
            if labels_func is not None:
                labels = labels_func(k)
            else:
                labels = self._get_labels(method, k, linkage_method, random_state)

            row = {
                "k": k,
                "silueta": self.silhouette(labels),
                "dunn": self.dunn(labels),
                "calinski_harabasz": self.calinski_harabasz(labels),
                "davies_bouldin": self.davies_bouldin(labels),
                "conectividad": self.connectivity(labels),
            }

            if compute_stability:
                row["prediction_strength"] = self.prediction_strength(
                    k, method=method, linkage_method=linkage_method,
                    random_state=random_state,
                )
                jac = self.bootstrap_jaccard(
                    labels, n_boot=n_boot, method=method,
                    linkage_method=linkage_method, random_state=random_state,
                )
                row["bootstrap_jaccard"] = jac["jaccard_global"]

            results.append(row)

        df = pd.DataFrame(results)
        label = f"{method}" + (f" + {linkage_method}" if method == "hierarchical" else "")
        print(f"\n=== Validación ({label}) ===")
        print(df.round(4).to_string(index=False))
        return df

    def add_gap(self, df, k_range, n_refs=20, random_state=42):
        gaps, sks = self.gap_statistic(k_range, n_refs=n_refs,
                                       random_state=random_state)
        df = df.copy()
        df["gap"] = gaps
        df["gap_s"] = sks
        print("\n--- Estadística Gap ---")
        for i, k in enumerate(df["k"]):
            print(f"k={k}: Gap={gaps[i]:.4f}, S_k={sks[i]:.4f}")
        return df

    def plot_silhouette_curve(self, df, method_label=""):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["k"], df["silueta"], "o-", color="steelblue", linewidth=2)
        best_k = df.loc[df["silueta"].idxmax(), "k"]
        ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.7,
                   label=f"Mejor k={int(best_k)}")
        ax.set_xlabel("Número de grupos (k)")
        ax.set_ylabel("Silueta promedio")
        ax.set_title(f"Silueta vs k {method_label}".strip())
        ax.set_xticks(df["k"])
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_silhouette_bars(self, labels, title="Silueta por individuo", save_path=None):
        if self.D is not None:
            sil_vals = silhouette_samples(self.D, labels, metric="precomputed")
        else:
            sil_vals = silhouette_samples(self.X, labels)
        n_clusters = len(np.unique(labels))
        avg_sil = sil_vals.mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 0
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

        for i, c in enumerate(sorted(np.unique(labels))):
            cluster_sil = np.sort(sil_vals[labels == c])
            size = cluster_sil.shape[0]
            y_upper = y_lower + size
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, cluster_sil,
                facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
            )
            ax.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=10)
            y_lower = y_upper + 10

        ax.axvline(x=avg_sil, color="red", linestyle="--",
                   label=f"Silueta media = {avg_sil:.4f}")
        ax.set_xlabel("Coeficiente de silueta")
        ax.set_ylabel("Cluster")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figura guardada: {save_path}")
        plt.show()

    def plot_stability_curve(self, df, method_label=""):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "prediction_strength" in df.columns:
            axes[0].plot(df["k"], df["prediction_strength"], "o-",
                         color="darkgreen", linewidth=2)
            axes[0].axhline(y=0.8, color="red", linestyle="--", alpha=0.7,
                            label="Umbral 0.8")
            axes[0].set_xlabel("k")
            axes[0].set_ylabel("Prediction Strength")
            axes[0].set_title(f"Prediction Strength {method_label}".strip())
            axes[0].set_xticks(df["k"])
            axes[0].legend()

        if "bootstrap_jaccard" in df.columns:
            axes[1].plot(df["k"], df["bootstrap_jaccard"], "o-",
                         color="darkorange", linewidth=2)
            axes[1].axhline(y=0.75, color="red", linestyle="--", alpha=0.7,
                            label="Umbral 0.75")
            axes[1].set_xlabel("k")
            axes[1].set_ylabel("Bootstrap Jaccard")
            axes[1].set_title(f"Bootstrap Jaccard {method_label}".strip())
            axes[1].set_xticks(df["k"])
            axes[1].legend()

        plt.tight_layout()
        plt.show()
