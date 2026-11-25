"""
Regenerate the README figures from the results committed in this repository.

  docs/figures/layer1_information_ceiling.png   species vs group accuracy (LOO-CV)
  docs/figures/layer2_domain_shift.png          extant-trained GMM log-likelihood
  docs/figures/layer3_latent_structure.png      BHC dendrogram + agreement with the taxonomy

docs/figures/pipeline.png is Figure 1 of thesis.pdf and is not generated here.

Usage (from the repository root):  python docs/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import BayesianGaussianMixture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

# Taxonomic groups used for the group-level task (thesis, Section 4.2).
GROUP = {
    "Papio anubis": "Papio", "Papio hamadryas": "Papio", "Papio ursinus": "Papio",
    "Cercocebus atys": "Mangabeys", "Lophocebus aterrimus": "Mangabeys",
    "Macaca sylvanus": "Macaca", "Theropithecus gelada": "Theropithecus",
}
SPECIES_ORDER = ["Papio anubis", "Papio hamadryas", "Papio ursinus", "Cercocebus atys",
                 "Lophocebus aterrimus", "Macaca sylvanus", "Theropithecus gelada"]
COLORS = {"Papio anubis": "#1f5f9e", "Papio hamadryas": "#4a90d9", "Papio ursinus": "#9cc3ec",
          "Cercocebus atys": "#d98b2b", "Lophocebus aterrimus": "#f2c078",
          "Macaca sylvanus": "#8e5ba8", "Theropithecus gelada": "#3f8f5b"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#888888", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def accuracy(path, level="species"):
    df = pd.read_csv(path)
    t, p = df["y_true"].str.strip(), df["y_pred_loo"].str.strip()
    if level == "group":
        t, p = t.map(GROUP), p.map(GROUP)
    return float((t == p).mean())


def layer1():
    blr = ROOT / "probabilistic_models" / "results" / "blr"
    hier = ROOT / "probabilistic_models" / "results" / "blr_hierarchical" / "D1"
    rows = [  # label, species accuracy, group accuracy
        ("Flat BLR · Gaussian", accuracy(blr / "gaussian/predicciones_loocv_blr_gaussian.csv"), None),
        ("Flat BLR · Laplace", accuracy(blr / "lasso/predicciones_loocv_blr_lasso.csv"), None),
        ("Flat BLR · horseshoe", accuracy(blr / "horseshoe/predicciones_loocv_blr_horseshoe.csv"), None),
        ("Flat BLR · spike-and-slab", accuracy(blr / "spike_and_slab/predicciones_loocv_blr_spike_and_slab.csv"), None),
        ("Hierarchical BLR · horseshoe",
         accuracy(hier / "predicciones_loocv_3niveles_horseshoe.csv"),
         accuracy(hier / "predicciones_loocv_3niveles_horseshoe.csv", "group")),
    ]
    truth = pd.read_csv(hier / "predicciones_loocv_3niveles_horseshoe.csv")["y_true"].str.strip()
    maj_species = truth.value_counts(normalize=True).iloc[0]
    maj_group = truth.map(GROUP).value_counts(normalize=True).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [5, 2.2]})
    ax = axes[0]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color="#4a78a8", height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(maj_species, color="#c0392b", ls="--", lw=1.2, label=f"majority class ({maj_species:.2f})")
    ax.axvline(1 / 7, color=MUTED, ls=":", lw=1.2, label=f"chance ({1/7:.2f})")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("LOO-CV accuracy")
    ax.set_title("Species level (7 classes): barely above the majority baseline")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

    ax = axes[1]
    sp, gr = rows[-1][1], rows[-1][2]
    ax.bar([0, 1], [sp, gr], color=["#9db7d3", "#1f5f9e"], width=0.6)
    ax.plot([-0.3, 0.3], [maj_species] * 2, color="#c0392b", ls="--", lw=1.2)
    ax.plot([0.7, 1.3], [maj_group] * 2, color="#c0392b", ls="--", lw=1.2)
    for xi, v in zip([0, 1], [sp, gr]):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["Species\n(7)", "Group\n(4)"])
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("LOO-CV accuracy")
    ax.set_title("Same model, two levels")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle("Layer 1 · The signal resolves the group, not the species "
                 "(dashed: majority-class baseline)", fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "layer1_information_ceiling.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {r[0]: (r[1], r[2]) for r in rows} | {"majority": (maj_species, maj_group)}


def layer2():
    ll = pd.read_csv(ROOT / "eda" / "results" / "gmm" / "nbool" / "verosimilitud_gmm_1.csv")
    ll["Estado"] = ll["Estado"].str.strip()
    ext = ll.loc[ll["Estado"] == "Extant", "Log_Likelihood"].values
    fos = ll.loc[ll["Estado"] != "Extant", "Log_Likelihood"].values

    fig, ax = plt.subplots(figsize=(10, 2.9))
    rng = np.random.default_rng(0)
    for yi, vals, col, name in [(1, ext, "#1f5f9e", f"Extant (n={len(ext)})"),
                                (0, fos, "#b5651d", f"Fossil (n={len(fos)})")]:
        ax.scatter(vals, yi + rng.uniform(-0.18, 0.18, len(vals)), s=14, alpha=0.7, color=col, lw=0)
        med = np.median(vals)
        ax.plot([med, med], [yi - 0.3, yi + 0.3], color=INK, lw=2)
        ax.text(med, yi + 0.36, f"median {med:,.3g}", ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("symlog", linthresh=100)
    ticks = [-1e7, -1e6, -1e5, -1e4, -1e3, 0, 1e3]
    ax.set_xticks(ticks, ["−10⁷", "−10⁶", "−10⁵", "−10⁴", "−10³", "0", "10³"])
    ax.set_yticks([0, 1], [f"Fossil (n={len(fos)})", f"Extant (n={len(ext)})"])
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("Log-likelihood under a Gaussian mixture fit on extant specimens (symlog scale)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Layer 2 · Fossils fall outside the extant distribution: a supervised classifier "
                 "would extrapolate (Mann–Whitney p < 10⁻¹⁸)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "layer2_domain_shift.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return float(np.median(ext)), float(np.median(fos))


def bhattacharyya(mu1, s1, mu2, s2):
    s = (s1 + s2) / 2
    d = mu1 - mu2
    return float(0.125 * d @ np.linalg.inv(s) @ d
                 + 0.5 * (np.linalg.slogdet(s)[1] - 0.5 * (np.linalg.slogdet(s1)[1] + np.linalg.slogdet(s2)[1])))


def layer3():
    df = pd.read_csv(ROOT / "visualization" / "gplvm" / "results_ld5_75" / "espacio_latente_gplvm.csv")
    lat = [c for c in df.columns if c.startswith("Latent_")]
    status = df["Extant/Extinct "].astype(str).str.strip()
    ext, fos = df[status == "Extant"], df[status == "Extinct"]
    X = ext[lat].values
    sp = ext["Target_Specie"].str.strip().values
    species = np.unique(sp)

    # Same procedure as visualization/gplvm/latent_structure_discovery.py
    dp = BayesianGaussianMixture(n_components=7, covariance_type="full",
                                 weight_concentration_prior_type="dirichlet_process",
                                 weight_concentration_prior=0.5, n_init=5, random_state=42,
                                 max_iter=500).fit(X)
    dp_labels = dp.predict(X)
    k = len(np.unique(dp_labels))
    stats = {}
    for s in species:
        pts = X[sp == s]
        stats[s] = (pts.mean(0), np.cov(pts, rowvar=False) + 1e-4 * np.eye(len(lat)))
    D = np.zeros((len(species), len(species)))
    for i, a in enumerate(species):
        for j, b in enumerate(species):
            if i != j:
                D[i, j] = bhattacharyya(*stats[a], *stats[b])
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    bhc = fcluster(Z, t=k, criterion="maxclust")
    bhc_specimen = np.array([bhc[np.where(species == s)[0][0]] for s in sp])
    scores = {
        "DP": (adjusted_rand_score(sp, dp_labels), normalized_mutual_info_score(sp, dp_labels)),
        "BHC": (adjusted_rand_score(sp, bhc_specimen), normalized_mutual_info_score(sp, bhc_specimen)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    dendrogram(Z, labels=list(species), ax=ax, leaf_rotation=35, leaf_font_size=9,
               color_threshold=0, above_threshold_color="#3f8f5b")
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.set_ylabel("Bhattacharyya distance")
    ax.set_title("Species clustered in the GPLVM latent space (BHC)", loc="left")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(2)
    for off, idx, name, col in [(-0.18, 0, "ARI", "#9cc9a8"), (0.18, 1, "NMI", "#3f8f5b")]:
        vals = [scores["DP"][idx], scores["BHC"][idx]]
        ax.bar(x + off, vals, width=0.36, color=col, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, max(v, 0) + 0.02, f"{abs(v):.2f}" if abs(v) < 0.005 else f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Dirichlet process\n(baseline)", "BHC"])
    ax.set_ylim(0, 1); ax.set_ylabel("Agreement with true species (1 = perfect)")
    ax.set_title("Agreement with the real taxonomy", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle("Layer 3 · Without using any label, the latent space recovers the phylogeny",
                 fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "layer3_latent_structure.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return scores


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    acc = layer1()
    med = layer2()
    sc = layer3()
    print("Layer 1 accuracy (species, group):")
    for k, v in acc.items():
        print(f"  {k:30s} {v[0]:.3f}  {'' if v[1] is None else f'{v[1]:.3f}'}")
    print(f"Layer 2 median log-likelihood: extant {med[0]:.4g}, fossil {med[1]:.4g}")
    print(f"Layer 3 ARI/NMI: BHC {sc['BHC'][0]:.3f}/{sc['BHC'][1]:.3f}, DP {sc['DP'][0]:.3f}/{sc['DP'][1]:.3f}")
    print(f"Figures written to {OUT}")
