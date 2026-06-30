import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.models.gplvm.latent_variable import VariationalLatentVariable
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# 1. PREPARACIÓN DE DATOS (Modificado para incluir fósiles en el espacio)
# =========================================================
FILE_PATH = "src/dataset_con_grupos.csv"
df = pd.read_csv(FILE_PATH)
metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']
feature_cols = [c for c in df.columns if c not in metadata_cols]

# Imputamos y escalamos TODOS los datos
X_raw = df[feature_cols].fillna(0)
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

# Usamos TODOS los datos para construir el mapa latente
Y_all_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)

# Máscaras y etiquetas (solo para colorear luego)
status = df['Extant/Extinct '].astype(str)
mask_extant = status.str.contains('Extant', case=False, na=False)
labels_species = df.loc[mask_extant, 'Target_Specie'].values

# =========================================================
# 2. MODELO CON INICIALIZACIÓN "JITTERED" (Rompe la simetría)
# =========================================================
def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q=latent_dim)
    X_pca = torch.matmul(Y, V[:, :latent_dim])
    # Añadimos un 10% de ruido para obligar al modelo a usar todas las dimensiones
    jitter = torch.randn_like(X_pca) * 0.1
    return torch.nn.Parameter(X_pca + jitter)

class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=True, Y=None):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        X_init = _init_pca(Y, latent_dim) if (pca and Y is not None) else torch.nn.Parameter(torch.randn(n, latent_dim))

        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        super().__init__(X, q_f)

        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, X):
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))

# =========================================================
# 3. ENTRENAMIENTO Y EXPORTACIÓN DEL ESPACIO LATENTE
# =========================================================
LATENT_DIM = 5  

N, data_dim = Y_all_tensor.shape
model = bGPLVM(N, data_dim, latent_dim=LATENT_DIM, n_inducing=75, Y=Y_all_tensor)
likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
likelihood.noise = 0.1
mll = VariationalELBO(likelihood, model, num_data=N)

optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.05)

print(f"\nEntrenando bGPLVM con {N} muestras (Extantes + Fósiles) en latent_dim={LATENT_DIM}...")
model.train()
likelihood.train()

for i in range(10000):
    optimizer.zero_grad()
    sample = model.sample_latent_variable()
    output = model(sample)
    loss = -mll(output, Y_all_tensor.T).sum()
    loss.backward()
    optimizer.step()
    if (i + 1) % 500 == 0:
        print(f"Iteración {i+1} - Loss: {loss.item():.3f}")

# Extraer coordenadas latentes de TODAS las muestras
X_all_mu = model.X.q_mu.detach().numpy()

# Separar la parte de entrenamiento (extantes) para las métricas de LDA/QDA
X_train_mu = X_all_mu[mask_extant.values]

# 💾 GUARDAR EL ESPACIO LATENTE EN CSV PARA EL MODELO JERÁRQUICO 💾
df_latent = df[metadata_cols].copy()
for d in range(LATENT_DIM):
    df_latent[f'Latent_{d+1}'] = X_all_mu[:, d]

csv_out = "espacio_latente_gplvm.csv"
df_latent.to_csv(csv_out, index=False)
print(f"\n✅ Espacio latente global guardado en: '{csv_out}'")

# =========================================================
# 4. VISUALIZACIÓN 2D (solo si latent_dim == 2)
# =========================================================
if LATENT_DIM == 2:
    plt.figure(figsize=(10, 10))
    unique_species = np.unique(labels_species)
    colors = sns.color_palette("husl", len(unique_species))

    for i, specie in enumerate(unique_species):
        idx = (labels_species == specie)
        plt.scatter(X_train_mu[idx, 0], X_train_mu[idx, 1],
                    c=[colors[i]], label=specie, alpha=0.7, edgecolors='w', s=80)

    plt.title('Mapa Ecológico Latente (Optimizado y Guardado)', fontsize=14)
    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='x-small')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("mapa_latente_especies.png", dpi=300, bbox_inches='tight')
    print("\n-> Imagen guardada como: 'mapa_latente_especies.png'")
    plt.show()
else:
    print(f"\n[INFO] Visualización 2D omitida (latent_dim={LATENT_DIM}). Solo se genera con latent_dim=2.")

# =========================================================
# 5. ANÁLISIS DE SEPARABILIDAD: LDA, QDA Y MATRIZ DE DISTANCIAS
# =========================================================

def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Distancia de Bhattacharyya entre dos distribuciones gaussianas.
    Mide la superposición entre clases: mayor valor = más separadas.
    """
    sigma_avg = (sigma1 + sigma2) / 2.0
    delta_mu = mu1 - mu2

    # Término de covarianza
    sign1, logdet_avg = np.linalg.slogdet(sigma_avg)
    sign2, logdet1   = np.linalg.slogdet(sigma1)
    sign3, logdet2   = np.linalg.slogdet(sigma2)

    if sign1 <= 0 or sign2 <= 0 or sign3 <= 0:
        # Fallback: distancia euclídea entre medias si las matrices son singulares
        return np.linalg.norm(delta_mu)

    cov_term = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))

    # Término de Mahalanobis
    try:
        sigma_avg_inv = np.linalg.inv(sigma_avg)
        maha_term = 0.125 * delta_mu @ sigma_avg_inv @ delta_mu
    except np.linalg.LinAlgError:
        maha_term = 0.125 * np.dot(delta_mu, delta_mu)

    return maha_term + cov_term


def pairwise_lda_qda_accuracy(X, labels):
    """
    Para cada par de especies, entrena LDA y QDA sobre solo esas dos clases
    y devuelve el accuracy LOO (Leave-One-Out).
    Requiere al menos 2 muestras por clase.
    """
    unique = np.unique(labels)
    n = len(unique)
    lda_mat = np.full((n, n), np.nan)
    qda_mat = np.full((n, n), np.nan)

    loo = LeaveOneOut()

    for i in range(n):
        for j in range(i + 1, n):
            mask = (labels == unique[i]) | (labels == unique[j])
            X_pair = X[mask]
            y_pair = labels[mask]

            # Necesitamos al menos 2 muestras por clase para LOO
            counts = {cls: np.sum(y_pair == cls) for cls in [unique[i], unique[j]]}
            if any(v < 2 for v in counts.values()):
                continue

            le = LabelEncoder()
            y_enc = le.fit_transform(y_pair)

            try:
                lda = LinearDiscriminantAnalysis()
                lda_scores = cross_val_score(lda, X_pair, y_enc, cv=loo)
                lda_acc = lda_scores.mean()
                lda_mat[i, j] = lda_acc
                lda_mat[j, i] = lda_acc
            except Exception:
                pass

            try:
                qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
                qda_scores = cross_val_score(qda, X_pair, y_enc, cv=loo)
                qda_acc = qda_scores.mean()
                qda_mat[i, j] = qda_acc
                qda_mat[j, i] = qda_acc
            except Exception:
                pass

    np.fill_diagonal(lda_mat, 1.0)
    np.fill_diagonal(qda_mat, 1.0)
    return unique, lda_mat, qda_mat


def pairwise_bhattacharyya(X, labels):
    """Matriz de distancias de Bhattacharyya entre especies."""
    unique = np.unique(labels)
    n = len(unique)
    dist_mat = np.zeros((n, n))

    species_stats = {}
    for sp in unique:
        pts = X[labels == sp]
        mu = pts.mean(axis=0)
        # Covarianza regularizada para evitar singularidades
        cov = np.cov(pts, rowvar=False) if pts.shape[0] > 1 else np.eye(X.shape[1])
        cov += np.eye(cov.shape[0]) * 1e-4
        species_stats[sp] = (mu, cov)

    for i, sp1 in enumerate(unique):
        for j, sp2 in enumerate(unique):
            if i == j:
                dist_mat[i, j] = 0.0
            elif i < j:
                d = bhattacharyya_distance(*species_stats[sp1], *species_stats[sp2])
                dist_mat[i, j] = d
                dist_mat[j, i] = d

    return unique, dist_mat


print("\nCalculando separabilidad LDA / QDA y distancias de Bhattacharyya...")

unique_sp, lda_mat, qda_mat = pairwise_lda_qda_accuracy(X_train_mu, labels_species)
_, bhat_mat = pairwise_bhattacharyya(X_train_mu, labels_species)

n_sp = len(unique_sp)
short_names = [s[:18] for s in unique_sp]  # Trunca nombres largos para el eje

# =========================================================
# 6. FIGURA: MATRIZ DE DISTANCIAS / SEPARABILIDAD
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 6))
fig.suptitle(
    f'Separabilidad entre Especies en Espacio Latente (latent_dim={LATENT_DIM})',
    fontsize=15, fontweight='bold', y=1.01
)

def plot_matrix(ax, mat, title, cmap, fmt=".2f", vmin=None, vmax=None, mask_diag=False):
    display = mat.copy().astype(float)
    if mask_diag:
        np.fill_diagonal(display, np.nan)

    # Máscara de NaN para celdas sin datos
    nan_mask = np.isnan(display)

    im = ax.imshow(display, cmap=cmap, aspect='auto',
                   vmin=vmin if vmin is not None else np.nanmin(display),
                   vmax=vmax if vmax is not None else np.nanmax(display))

    # Valores en cada celda
    for r in range(n_sp):
        for c in range(n_sp):
            if nan_mask[r, c]:
                ax.text(c, r, "—", ha='center', va='center', fontsize=7, color='gray')
            else:
                val = display[r, c]
                ax.text(c, r, format(val, fmt), ha='center', va='center',
                        fontsize=7, color='white' if abs(val) > (np.nanmax(display) * 0.6) else 'black')

    ax.set_xticks(range(n_sp))
    ax.set_yticks(range(n_sp))
    ax.set_xticklabels(short_names, rotation=60, ha='right', fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title(title, fontsize=12, pad=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# LDA accuracy (0→0.5 = no separable, 1.0 = perfectamente separable)
plot_matrix(axes[0], lda_mat, 'LDA Accuracy (LOO)\n[mayor = más separable]',
            cmap='RdYlGn', fmt=".2f", vmin=0.0, vmax=1.0)

# QDA accuracy
plot_matrix(axes[1], qda_mat, 'QDA Accuracy (LOO)\n[mayor = más separable]',
            cmap='RdYlGn', fmt=".2f", vmin=0.0, vmax=1.0)

# Bhattacharyya distance (0 = idénticas, >1 = muy separadas)
plot_matrix(axes[2], bhat_mat, 'Distancia de Bhattacharyya\n[mayor = más separadas]',
            cmap='Blues', fmt=".2f", mask_diag=True)

plt.tight_layout()
out_path = f"separabilidad_especies_ld{LATENT_DIM}.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\n-> Imagen guardada como: '{out_path}'")
plt.show()

# =========================================================
# 7. RESUMEN GLOBAL LDA / QDA (todas las clases a la vez)
# =========================================================
print("\n--- Clasificación global (todas las especies) ---")
le_global = LabelEncoder()
y_global = le_global.fit_transform(labels_species)

# LDA global (solo tiene sentido si n_samples >> n_classes)
try:
    lda_global = LinearDiscriminantAnalysis()
    lda_global_scores = cross_val_score(lda_global, X_train_mu, y_global,
                                        cv=min(5, N), scoring='accuracy')
    print(f"LDA global  — Accuracy CV: {lda_global_scores.mean():.3f} ± {lda_global_scores.std():.3f}")
except Exception as e:
    print(f"LDA global  — No disponible: {e}")

try:
    qda_global = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda_global_scores = cross_val_score(qda_global, X_train_mu, y_global,
                                        cv=min(5, N), scoring='accuracy')
    print(f"QDA global  — Accuracy CV: {qda_global_scores.mean():.3f} ± {qda_global_scores.std():.3f}")
except Exception as e:
    print(f"QDA global  — No disponible: {e}")


# =========================================================
# 8. DISTANCIAS INDIVIDUO → ESPECIE (FÓSILES Y EXTANTES)
#    Calcula distancia euclídea y de Mahalanobis de cada
#    espécimen al centroide de cada especie extante.
# =========================================================
print("\n--- Distancias individuo → especie (espacio latente) ---")

# Centroides de cada especie extante
centroids = {}
cov_matrices = {}
for sp in unique_sp:
    pts = X_train_mu[labels_species == sp]
    centroids[sp] = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False) if pts.shape[0] > 1 else np.eye(X_train_mu.shape[1])
    cov_matrices[sp] = cov + np.eye(cov.shape[0]) * 1e-4   # regularización

centroid_matrix = np.stack([centroids[sp] for sp in unique_sp])  # (n_sp, latent_dim)

# --- Distancias euclídeas de TODOS los especímenes a cada centroide ---
D_euclidean = cdist(X_all_mu, centroid_matrix, metric='euclidean')   # (N, n_sp)

df_distances = df[metadata_cols].copy()
for k, sp in enumerate(unique_sp):
    df_distances[f'dist_eucl_{sp[:20]}'] = D_euclidean[:, k]

df_distances['nearest_species_eucl'] = [unique_sp[i] for i in D_euclidean.argmin(axis=1)]
df_distances['min_dist_eucl']        = D_euclidean.min(axis=1)

csv_dist = "distancias_especimenes_especies.csv"
df_distances.to_csv(csv_dist, index=False)
print(f"✅ Tabla de distancias guardada en: '{csv_dist}'")

# --- Visualización: heatmap de distancias (solo fósiles para no saturar) ---
mask_extinct = ~mask_extant.values
if mask_extinct.sum() > 0:
    D_fossils = D_euclidean[mask_extinct]
    fossil_labels = df.loc[mask_extinct, 'Specimen'].astype(str).values

    fig_d, ax_d = plt.subplots(figsize=(max(8, len(unique_sp) * 0.7),
                                        max(6, mask_extinct.sum() * 0.35)))
    im_d = ax_d.imshow(D_fossils, cmap='viridis_r', aspect='auto')
    ax_d.set_xticks(range(len(unique_sp)))
    ax_d.set_xticklabels([s[:15] for s in unique_sp], rotation=60, ha='right', fontsize=7)
    ax_d.set_yticks(range(len(fossil_labels)))
    ax_d.set_yticklabels(fossil_labels, fontsize=7)
    ax_d.set_title('Distancia Euclídea Fósil → Especie Extante (espacio latente)',
                   fontsize=12, fontweight='bold')
    ax_d.set_xlabel('Especie extante')
    ax_d.set_ylabel('Fósil / Espécimen extinto')
    plt.colorbar(im_d, ax=ax_d, label='Distancia euclídea')

    # Marca el mínimo de cada fila (especie más cercana)
    for r in range(D_fossils.shape[0]):
        best_col = D_fossils[r].argmin()
        ax_d.add_patch(plt.Rectangle((best_col - 0.5, r - 0.5), 1, 1,
                                      fill=False, edgecolor='red', lw=1.5))

    plt.tight_layout()
    plt.savefig("distancias_fosiles_especies.png", dpi=300, bbox_inches='tight')
    print("-> Imagen guardada como: 'distancias_fosiles_especies.png'")
    plt.show()
else:
    print("[INFO] No se detectaron fósiles/extintos; se omite el heatmap de distancias.")


# =========================================================
# 9. INCERTIDUMBRE VARIACIONAL (q_var)
#    Cada punto del BGPLVM tiene una nube de incertidumbre.
#    Se compara la incertidumbre media entre extantes y fósiles.
# =========================================================
print("\n--- Incertidumbre variacional (q_sigma) del espacio latente ---")

# q_log_sigma → sigma por espécimen y dimensión latente
X_sigma = model.X.q_log_sigma.exp().detach().numpy()   # (N, latent_dim)

df_uncertainty = df[metadata_cols].copy()
df_uncertainty['sigma_mean']   = X_sigma.mean(axis=1)
df_uncertainty['sigma_max']    = X_sigma.max(axis=1)
for d in range(LATENT_DIM):
    df_uncertainty[f'sigma_ld{d+1}'] = X_sigma[:, d]

csv_unc = "incertidumbre_variacional.csv"
df_uncertainty.to_csv(csv_unc, index=False)
print(f"✅ Tabla de incertidumbre guardada en: '{csv_unc}'")

# Comparación extantes vs fósiles
sigma_extant  = X_sigma[mask_extant.values].mean(axis=1)
sigma_extinct = X_sigma[mask_extinct].mean(axis=1) if mask_extinct.sum() > 0 else np.array([])

print(f"  Sigma media — Extantes : {sigma_extant.mean():.4f} ± {sigma_extant.std():.4f}")
if sigma_extinct.size > 0:
    print(f"  Sigma media — Fósiles  : {sigma_extinct.mean():.4f} ± {sigma_extinct.std():.4f}")

# --- Visualización: incertidumbre por dimensión latente (boxplot) ---
fig_u, axes_u = plt.subplots(1, 2, figsize=(14, 5))
fig_u.suptitle('Incertidumbre Variacional en el Espacio Latente', fontsize=13, fontweight='bold')

# Boxplot por dimensión latente (extantes vs fósiles)
ax_box = axes_u[0]
positions_e = np.arange(LATENT_DIM) - 0.18
positions_f = np.arange(LATENT_DIM) + 0.18

bp_e = ax_box.boxplot(
    [X_sigma[mask_extant.values, d] for d in range(LATENT_DIM)],
    positions=positions_e, widths=0.3, patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6), medianprops=dict(color='navy'),
    whiskerprops=dict(color='steelblue'), capprops=dict(color='steelblue'),
    flierprops=dict(marker='o', markersize=3, color='steelblue', alpha=0.4)
)

if mask_extinct.sum() > 0:
    bp_f = ax_box.boxplot(
        [X_sigma[mask_extinct, d] for d in range(LATENT_DIM)],
        positions=positions_f, widths=0.3, patch_artist=True,
        boxprops=dict(facecolor='tomato', alpha=0.6), medianprops=dict(color='darkred'),
        whiskerprops=dict(color='tomato'), capprops=dict(color='tomato'),
        flierprops=dict(marker='o', markersize=3, color='tomato', alpha=0.4)
    )
    ax_box.legend([bp_e['boxes'][0], bp_f['boxes'][0]], ['Extantes', 'Fósiles/Extintos'],
                  fontsize=9)
else:
    ax_box.legend([bp_e['boxes'][0]], ['Extantes'], fontsize=9)

ax_box.set_xticks(range(LATENT_DIM))
ax_box.set_xticklabels([f'LD{d+1}' for d in range(LATENT_DIM)])
ax_box.set_ylabel('σ variacional')
ax_box.set_title('σ por dimensión latente')
ax_box.grid(True, linestyle='--', alpha=0.4)

# Scatter: sigma_mean vs nearest_species distance (solo fósiles)
# Sin nombres de individuos; cada punto se codifica por su especie extante
# más cercana mediante la FORMA del marcador (el color sigue siendo σ media).
ax_sc = axes_u[1]
if mask_extinct.sum() > 0:
    y_sc = D_euclidean[mask_extinct].min(axis=1)
    # Especie extante más cercana de cada fósil (índice -> nombre)
    nearest_idx = D_euclidean[mask_extinct].argmin(axis=1)

    # Repertorio de formas; se reparte entre las especies presentes
    markers_pool = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'p', 'h']
    present_sp = sorted(set(nearest_idx))
    marker_for_sp = {sp_i: markers_pool[k % len(markers_pool)]
                     for k, sp_i in enumerate(present_sp)}

    sc = None
    for sp_i in present_sp:
        m = nearest_idx == sp_i
        sc = ax_sc.scatter(
            sigma_extinct[m], y_sc[m],
            c=sigma_extinct[m], cmap='Reds',
            vmin=sigma_extinct.min(), vmax=sigma_extinct.max(),
            marker=marker_for_sp[sp_i], s=70,
            edgecolors='k', linewidths=0.5, zorder=3
        )
    plt.colorbar(sc, ax=ax_sc, label='σ media')

    # Leyenda de formas -> especie (marcador en negro para que se distinga
    # de la escala de color, que codifica σ).
    shape_handles = [
        plt.Line2D([0], [0], marker=marker_for_sp[sp_i], color='none',
                   markerfacecolor='gray', markeredgecolor='k',
                   markersize=8, linestyle='None', label=unique_sp[sp_i][:18])
        for sp_i in present_sp
    ]
    ax_sc.legend(handles=shape_handles, title='Especie extante más cercana',
                 fontsize=7, title_fontsize=8, loc='best')

    ax_sc.set_xlabel('Incertidumbre media (σ)')
    ax_sc.set_ylabel('Distancia mín. a especie extante')
    ax_sc.set_title('Fósiles: incertidumbre vs distancia a especie más cercana')
    ax_sc.grid(True, linestyle='--', alpha=0.4)
else:
    ax_sc.text(0.5, 0.5, 'Sin fósiles detectados', ha='center', va='center',
               transform=ax_sc.transAxes, fontsize=12, color='gray')
    ax_sc.set_title('Sin fósiles detectados')

plt.tight_layout()
plt.savefig("incertidumbre_variacional.png", dpi=300, bbox_inches='tight')
print("-> Imagen guardada como: 'incertidumbre_variacional.png'")
plt.show()


# =========================================================
# 10. DENDROGRAMA JERÁRQUICO SOBRE DISTANCIAS DE BHATTACHARYYA
#     Agrupa especies por similitud en el espacio latente.
# =========================================================
print("\n--- Dendrograma jerárquico (distancias de Bhattacharyya) ---")

# Convertir la matriz simétrica a forma condensada para scipy
from scipy.spatial.distance import squareform
bhat_condensed = squareform(bhat_mat, checks=False)

# Linkage con método 'average' (UPGMA), estándar en filogenética
Z_dendro = linkage(bhat_condensed, method='average')

n_species = len(unique_sp)
fig_h, ax_h = plt.subplots(figsize=(max(10, n_species * 0.55), 6))

dend = dendrogram(
    Z_dendro,
    labels=unique_sp,
    ax=ax_h,
    leaf_rotation=60,
    leaf_font_size=8,
    color_threshold=0.7 * max(Z_dendro[:, 2]),   # colorea clusters principales
)

ax_h.set_title(
    f'Dendrograma Jerárquico de Especies\n(distancia de Bhattacharyya, espacio latente {LATENT_DIM}D, UPGMA)',
    fontsize=12, fontweight='bold'
)
ax_h.set_xlabel('Especie', fontsize=10)
ax_h.set_ylabel('Distancia de Bhattacharyya', fontsize=10)
ax_h.grid(True, axis='y', linestyle='--', alpha=0.35)

plt.tight_layout()
out_dendro = f"dendrograma_especies_ld{LATENT_DIM}.png"
plt.savefig(out_dendro, dpi=300, bbox_inches='tight')
print(f"✅ Dendrograma guardado como: '{out_dendro}'")
plt.show()

print("\n✅ Análisis completo finalizado.")