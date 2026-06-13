"""
Wine Dataset Analysis
=====================
Task A: Data Preprocessing & Preliminary Statistical Analysis
Task B: Classification (SVM, kNN) -> Accuracy, Kappa, ROC-AUC
Task C: Clustering (k-Means, DBSCAN) -> Entropy, SSE

Dataset: UCI Wine Recognition Dataset (178 samples, 13 numeric features, 3 classes)
Source : https://archive.ics.uci.edu/ml/datasets/Wine
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report,
    silhouette_score
)
from scipy.stats import entropy as scipy_entropy

sns.set_theme(style="whitegrid")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42


# =====================================================================
# TASK A: DATA LOADING, PREPROCESSING & PRELIMINARY ANALYSIS
# =====================================================================

def load_data():
    raw = load_wine()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"] = raw.target
    target_names = list(raw.target_names)
    return df, target_names


def preliminary_analysis(df, target_names):
    print("=" * 70)
    print("TASK A: PRELIMINARY ANALYSIS")
    print("=" * 70)

    print(f"\nShape (rows, cols): {df.shape}")
    print(f"Rows : {df.shape[0]}")
    print(f"Cols : {df.shape[1]} (13 features + 1 target)")
    print(f"\nFeatures:\n{list(df.columns[:-1])}")
    print(f"\nClass distribution:")
    for i, name in enumerate(target_names):
        print(f"  Class {i} ({name}): {(df['target'] == i).sum()} samples")

    print(f"\nMissing values per column:\n{df.isnull().sum().sum()} total missing values")

    print("\nDescriptive statistics:")
    desc = df.describe().T
    print(desc[["mean", "std", "min", "max"]])

    # Save summary stats to CSV
    desc.to_csv(os.path.join(DATA_DIR, "descriptive_stats.csv"))

    # Correlation matrix
    corr = df.drop(columns=["target"]).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap (Wine Dataset)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_correlation_heatmap.png"), dpi=120)
    plt.close()

    # Class distribution bar chart
    plt.figure(figsize=(6, 5))
    counts = df["target"].value_counts().sort_index()
    bars = plt.bar([target_names[i] for i in counts.index], counts.values,
                    color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Class Distribution")
    plt.ylabel("Number of Samples")
    for b, c in zip(bars, counts.values):
        plt.text(b.get_x() + b.get_width()/2, c + 1, str(c), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_class_distribution.png"), dpi=120)
    plt.close()

    # Pie chart of class distribution
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=[target_names[i] for i in counts.index],
            autopct="%1.1f%%", colors=["#4C72B0", "#DD8452", "#55A868"],
            startangle=90)
    plt.title("Class Distribution (Proportion)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_class_pie.png"), dpi=120)
    plt.close()

    # Boxplots of a few key features by class
    key_features = ["alcohol", "flavanoids", "color_intensity", "proline"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat in zip(axes.flatten(), key_features):
        sns.boxplot(x="target", y=feat, data=df, ax=ax, hue="target", palette="Set2", legend=False)
        ax.set_xticks(range(len(target_names)))
        ax.set_xticklabels(target_names)
        ax.set_title(f"{feat} by class")
    plt.suptitle("Feature Distributions by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_boxplots_by_class.png"), dpi=120)
    plt.close()

    # Histograms of all features
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for ax, col in zip(axes.flatten(), df.columns[:-1]):
        ax.hist(df[col], bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_title(col, fontsize=9)
    for ax in axes.flatten()[len(df.columns)-1:]:
        ax.axis("off")
    plt.suptitle("Feature Distributions (Histograms)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_feature_histograms.png"), dpi=120)
    plt.close()

    return corr


# =====================================================================
# TASK B: CLASSIFICATION (SVM, kNN)
# =====================================================================

def run_classification(df, target_names):
    print("\n" + "=" * 70)
    print("TASK B: CLASSIFICATION (SVM vs kNN)")
    print("=" * 70)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # Standardize features (essential for SVM and kNN - distance-based)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    results = {}

    # ---- SVM ----
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_proba_svm = svm.predict_proba(X_test)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    kappa_svm = cohen_kappa_score(y_test, y_pred_svm)
    roc_svm = roc_auc_score(y_test, y_proba_svm, multi_class="ovr")

    results["SVM"] = {
        "accuracy": acc_svm, "kappa": kappa_svm, "roc_auc": roc_svm,
        "y_pred": y_pred_svm, "y_proba": y_proba_svm
    }

    print(f"\n--- SVM (RBF kernel) ---")
    print(f"Accuracy : {acc_svm:.4f}")
    print(f"Kappa    : {kappa_svm:.4f}")
    print(f"ROC-AUC  : {roc_svm:.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=target_names))

    # ---- kNN ----
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_proba_knn = knn.predict_proba(X_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    kappa_knn = cohen_kappa_score(y_test, y_pred_knn)
    roc_knn = roc_auc_score(y_test, y_proba_knn, multi_class="ovr")

    results["kNN"] = {
        "accuracy": acc_knn, "kappa": kappa_knn, "roc_auc": roc_knn,
        "y_pred": y_pred_knn, "y_proba": y_proba_knn
    }

    print(f"\n--- kNN (k=5) ---")
    print(f"Accuracy : {acc_knn:.4f}")
    print(f"Kappa    : {kappa_knn:.4f}")
    print(f"ROC-AUC  : {roc_knn:.4f}")
    print(classification_report(y_test, y_pred_knn, target_names=target_names))

    # ---- Comparison bar chart ----
    metrics = ["accuracy", "kappa", "roc_auc"]
    svm_vals = [results["SVM"][m] for m in metrics]
    knn_vals = [results["kNN"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, svm_vals, width, label="SVM", color="#4C72B0")
    plt.bar(x + width/2, knn_vals, width, label="kNN", color="#DD8452")
    plt.xticks(x, ["Accuracy", "Kappa", "ROC-AUC"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("SVM vs kNN: Classification Metrics Comparison")
    for i, (s, k) in enumerate(zip(svm_vals, knn_vals)):
        plt.text(i - width/2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
        plt.text(i + width/2, k + 0.01, f"{k:.3f}", ha="center", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_classification_comparison.png"), dpi=120)
    plt.close()

    # ---- Confusion matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, [("SVM", results["SVM"]), ("kNN", results["kNN"])]):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_confusion_matrices.png"), dpi=120)
    plt.close()

    # ---- ROC curves (one-vs-rest, per class) ----
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    from sklearn.metrics import roc_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, res) in zip(axes, [("SVM", results["SVM"]), ("kNN", results["kNN"])]):
        for i, cls_name in enumerate(target_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], res["y_proba"][:, i])
            roc_auc_i = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc_i:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_roc_curves.png"), dpi=120)
    plt.close()

    return results, X_scaled, y


# =====================================================================
# TASK C: CLUSTERING (k-Means, DBSCAN)
# =====================================================================

def cluster_entropy(y_true, cluster_labels):
    """
    Average entropy of class labels within each cluster, weighted by
    cluster size. Lower entropy = clusters are more "pure" w.r.t. true class.
    Noise points (DBSCAN label = -1) are treated as their own group.
    """
    df_tmp = pd.DataFrame({"true": y_true, "cluster": cluster_labels})
    total = len(df_tmp)
    weighted_entropy = 0.0
    for c in np.unique(cluster_labels):
        subset = df_tmp[df_tmp["cluster"] == c]["true"]
        counts = subset.value_counts().values
        probs = counts / counts.sum()
        e = scipy_entropy(probs, base=2)
        weighted_entropy += (len(subset) / total) * e
    return weighted_entropy


def sse_score(X, labels, centers=None):
    """Sum of Squared Errors (within-cluster sum of squares)."""
    sse = 0.0
    for c in np.unique(labels):
        if c == -1:
            continue  # skip noise for SSE
        pts = X[labels == c]
        if centers is not None and c < len(centers):
            center = centers[c]
        else:
            center = pts.mean(axis=0)
        sse += ((pts - center) ** 2).sum()
    return sse


def run_clustering(X_scaled, y, target_names):
    print("\n" + "=" * 70)
    print("TASK C: CLUSTERING (k-Means vs DBSCAN)")
    print("=" * 70)

    results = {}

    # ---- k-distance plot to justify DBSCAN eps ----
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dist_sorted = np.sort(distances[:, -1])

    plt.figure(figsize=(7, 5))
    plt.plot(k_dist_sorted, color="#4C72B0")
    plt.axhline(y=2.2, color="red", linestyle="--", alpha=0.6, label="eps = 2.2 (chosen)")
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel("3rd Nearest Neighbour Distance")
    plt.title("k-Distance Plot for DBSCAN eps Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08b_kdistance_plot.png"), dpi=120)
    plt.close()

    # ---- k-Means ----
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    km_sse = sse_score(X_scaled, km_labels, kmeans.cluster_centers_)
    km_entropy = cluster_entropy(y, km_labels)
    km_sil = silhouette_score(X_scaled, km_labels)

    results["KMeans"] = {
        "labels": km_labels, "sse": km_sse, "entropy": km_entropy,
        "silhouette": km_sil, "n_clusters": 3
    }

    print(f"\n--- k-Means (k=3) ---")
    print(f"SSE          : {km_sse:.3f}")
    print(f"Entropy      : {km_entropy:.4f} bits")
    print(f"Silhouette   : {km_sil:.4f}")
    print(f"Cluster sizes: {np.bincount(km_labels)}")

    # ---- DBSCAN ----
    # eps tuned via k-distance plot (5-NN) for standardized 13-D wine data
    dbscan = DBSCAN(eps=2.2, min_samples=3)
    db_labels = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    db_sse = sse_score(X_scaled, db_labels)
    db_entropy = cluster_entropy(y, db_labels)

    db_sil = None
    if n_clusters_db > 1:
        mask = db_labels != -1
        if mask.sum() > 1 and len(set(db_labels[mask])) > 1:
            db_sil = silhouette_score(X_scaled[mask], db_labels[mask])

    results["DBSCAN"] = {
        "labels": db_labels, "sse": db_sse, "entropy": db_entropy,
        "silhouette": db_sil, "n_clusters": n_clusters_db, "n_noise": n_noise
    }

    print(f"\n--- DBSCAN (eps=2.2, min_samples=3) ---")
    print(f"Clusters found : {n_clusters_db}")
    print(f"Noise points   : {n_noise}")
    print(f"SSE            : {db_sse:.3f}")
    print(f"Entropy        : {db_entropy:.4f} bits")
    if db_sil is not None:
        print(f"Silhouette     : {db_sil:.4f}")

    # ---- Comparison bar chart: SSE & Entropy ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    algos = ["k-Means", "DBSCAN"]
    sse_vals = [km_sse, db_sse]
    ent_vals = [km_entropy, db_entropy]

    axes[0].bar(algos, sse_vals, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("SSE Comparison (lower = tighter clusters)")
    axes[0].set_ylabel("SSE")
    for i, v in enumerate(sse_vals):
        axes[0].text(i, v + max(sse_vals)*0.01, f"{v:.1f}", ha="center")

    axes[1].bar(algos, ent_vals, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Entropy Comparison (lower = purer clusters)")
    axes[1].set_ylabel("Entropy (bits)")
    for i, v in enumerate(ent_vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_clustering_comparison.png"), dpi=120)
    plt.close()

    # ---- PCA visualization of clusters vs true labels ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    sc0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set2", s=40, edgecolor="k", linewidth=0.3)
    axes[0].set_title("True Class Labels")
    axes[0].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=n,
               markerfacecolor=sc0.cmap(sc0.norm(i)), markersize=8) for i, n in enumerate(target_names)]
    axes[0].legend(handles=handles)

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="Set2", s=40, edgecolor="k", linewidth=0.3)
    axes[1].set_title("k-Means Clusters")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="tab10", s=40, edgecolor="k", linewidth=0.3)
    axes[2].set_title(f"DBSCAN Clusters ({n_clusters_db} clusters, {n_noise} noise pts)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")

    plt.suptitle("PCA Projection: True Labels vs Cluster Assignments")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_pca_clusters.png"), dpi=120)
    plt.close()

    # ---- Elbow plot for k selection (justifies k=3) ----
    sse_list = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        sse_list.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(list(K_range), sse_list, "o-", color="#4C72B0")
    plt.axvline(x=3, color="red", linestyle="--", alpha=0.6, label="k=3 (chosen)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(list(K_range))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "11_elbow_method.png"), dpi=120)
    plt.close()

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    df, target_names = load_data()
    df.to_csv(os.path.join(DATA_DIR, "wine_dataset.csv"), index=False)

    corr = preliminary_analysis(df, target_names)
    clf_results, X_scaled, y = run_classification(df, target_names)
    clust_results = run_clustering(X_scaled, y, target_names)

    # Save final summary as JSON for reuse in report
    summary = {
        "dataset": {
            "name": "UCI Wine Recognition Dataset",
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1),
            "classes": target_names,
            "class_counts": {target_names[i]: int((df["target"] == i).sum()) for i in range(3)}
        },
        "classification": {
            "SVM": {
                "accuracy": float(clf_results["SVM"]["accuracy"]),
                "kappa": float(clf_results["SVM"]["kappa"]),
                "roc_auc": float(clf_results["SVM"]["roc_auc"]),
            },
            "kNN": {
                "accuracy": float(clf_results["kNN"]["accuracy"]),
                "kappa": float(clf_results["kNN"]["kappa"]),
                "roc_auc": float(clf_results["kNN"]["roc_auc"]),
            }
        },
        "clustering": {
            "KMeans": {
                "sse": float(clust_results["KMeans"]["sse"]),
                "entropy": float(clust_results["KMeans"]["entropy"]),
                "silhouette": float(clust_results["KMeans"]["silhouette"]),
            },
            "DBSCAN": {
                "sse": float(clust_results["DBSCAN"]["sse"]),
                "entropy": float(clust_results["DBSCAN"]["entropy"]),
                "silhouette": float(clust_results["DBSCAN"]["silhouette"]) if clust_results["DBSCAN"]["silhouette"] is not None else None,
                "n_clusters": int(clust_results["DBSCAN"]["n_clusters"]),
                "n_noise": int(clust_results["DBSCAN"]["n_noise"]),
            }
        }
    }

    with open(os.path.join(DATA_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE. Plots saved to:", PLOT_DIR)
    print("Summary saved to   :", os.path.join(DATA_DIR, "results_summary.json"))
    print("=" * 70)


if __name__ == "__main__":
    main()
