import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import scipy.stats
from pathlib import Path


def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(
        data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Get the PCA loadings
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return principal_df, pca_loadings


def perform_lda(X, y, n_components=None):
    if n_components is None or n_components > len(np.unique(y)) - 1:
        n_components = len(np.unique(y)) - 1

    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X, y)
    lda_df = pd.DataFrame(
        data=lda_components, columns=[f"LD{i+1}" for i in range(n_components)]
    )

    # Get the LDA coefficients
    lda_coefficients = pd.DataFrame(
        lda.scalings_,
        index=X.columns,
        columns=[f"LD{i+1}" for i in range(n_components)],
    )

    return lda_df, lda_coefficients


def perform_umap(df, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    embedding_umap = reducer.fit_transform(df)
    umap_df = pd.DataFrame(
        data=embedding_umap, columns=[f"UMAP{i+1}" for i in range(n_components)]
    )
    return umap_df


def perform_tsne(df, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    embedding_tsne = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(
        data=embedding_tsne, columns=[f"tSNE{i+1}" for i in range(n_components)]
    )
    return tsne_df


def plot_and_save(df, axe, hue_col, title, filename):
    plt.figure(figsize=(8, 6))

    # Fit and plot Gaussian for each class
    classes = df[hue_col].unique()

    x_col = axe + "1"
    y_col = axe + "2"

    # Seaborn palette
    # colors = sns.color_palette('Set2', len(classes))

    # Get the default color cycle from matplotlib
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Define a list of markers to use for each class
    markers = ["o", "s", "D", "^", "v", "P", "*"]  # Add more markers if needed

    for cls, color, marker in zip(classes, colors, markers):
        class_data = df[df[hue_col] == cls]
        mean = class_data[[x_col, y_col]].mean().values
        cov = np.cov(class_data[[x_col, y_col]].values.T)

        # Plot the data and the Gaussian ellipse
        plt.scatter(
            class_data[x_col],
            class_data[y_col],
            color=color,
            marker=marker,
            label=f"{cls}",
            alpha=0.7,
            edgecolor="k",
        )
        plot_cov_ellipse(
            cov,
            mean,
            nstd=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.2,
            fill=True,
            linewidth=2,
        )

    # sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='Set2')

    plt.title(title)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend(
        title=hue_col, bbox_to_anchor=(1, 0), loc="lower right", frameon=False
    )  #  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    print(f" - Plot saved to {filename}")
    plt.close()


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Plots an ellipse representing the covariance matrix cov centered at pos.
    Parameters:
    cov : 2x2 covariance matrix
    pos : 2-element array-like (mean of the distribution)
    nstd : number of standard deviations (default 2 standard deviations)
    kwargs : additional arguments passed to the ellipse patch
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(eigvals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax = plt.gca()
    ax.add_patch(ellip)
    return ellip


def process_and_plot(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "pca",  # "pca" or "lda"
    *,
    subpath: str = "",
    path_plots: str,
    path_coeffs: str,
):
    class_col = y.name

    # --- Paths ---
    path_plots = Path(path_plots)
    path_coeffs = Path(path_coeffs)
    if subpath:  # if subpath != "":
        subpath = Path(subpath)
        path_plots /= subpath
        path_coeffs /= subpath
    path_plots.mkdir(parents=True, exist_ok=True)
    path_coeffs.mkdir(parents=True, exist_ok=True)

    # Keep only numeric columns (coerce -> NaN, then drop columns that become all-NaN)
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    # --- Dimensionality Reduction: method-specific config ---
    methods = {
        "pca": {
            "run": lambda X, y: perform_pca(X),
            "axe": "PC",
            "title": "PCA Result",
        },
        "lda": {
            "run": lambda X, y: perform_lda(X, y),
            "axe": "LD",
            "title": "LDA Result",
        },
    }
    if method not in methods:
        raise ValueError("Unsupported method. Use 'pca' or 'lda'.")

    plots = []  # [(df, axe, title, filename), ...]

    reducer = methods[method]
    df_red, coefficients = reducer["run"](X, y)
    # Same as doing reduction_df[class_col] = y
    df_red = df_red.assign(**{class_col: y})
    plot_and_save(
        df_red,
        reducer["axe"],
        class_col,
        reducer["title"],
        path_plots / f"{method}_red.png",
    )

    # --- Embeddings on reduced components and raw data ---
    red_2d = df_red.loc[:, [reducer["axe"] + "1", reducer["axe"] + "2"]]
    df_red_umap = perform_umap(red_2d).assign(**{class_col: y})
    plot_and_save(
        df_red_umap,
        "UMAP",
        class_col,
        "UMAP on Reduced Components",
        path_plots / f"{method}_red_umap.png",
    )

    df_red_tsne = perform_tsne(red_2d).assign(**{class_col: y})
    plot_and_save(
        df_red_tsne,
        "tSNE",
        class_col,
        "t-SNE on Reduced Components",
        path_plots / f"{method}_red_tsne.png",
    )
    df_raw_umap = perform_umap(X).assign(**{class_col: y})
    plot_and_save(
        df_raw_umap, "UMAP", class_col, "UMAP on Raw Data", path_plots / "umap_raw.png"
    )
    df_raw_tsne = perform_tsne(X).assign(**{class_col: y})
    plot_and_save(
        df_raw_tsne, "tSNE", class_col, "t-SNE on Raw Data", path_plots / "tsne_raw.png"
    )

    # --- Plot and save figures ---
    saved_paths = []
    for plot_df, axe, title, filename in plots:
        saved_paths.append(path_plots / filename)

    # --- Save coefficients ---
    coefficients_filename = path_coeffs / f"{method}_coeffs.csv"
    coefficients.to_csv(coefficients_filename)
    print(f"\nCoefficients saved to {coefficients_filename}")
