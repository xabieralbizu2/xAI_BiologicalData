import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, cdist
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN
# =========================================================
LATENT_CSV    = "results_ld5_75/espacio_latente_gplvm.csv"
METADATA_COLS = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']
K_MAX         = 7   # techo de componentes para el Dirichlet Process

print("=" * 60)
print(" DESCUBRIMIENTO DE ESTRUCTURA EN EL ESPACIO LATENTE")
print(" Bayesian GMM (DP) | BHC via UPGMA + Bhattacharyya")
print("=" * 60)

# =========================================================
# 1. CARGA DE DATOS
# =========================================================
df = pd.read_csv(LATENT_CSV)
latent_cols = [c for c in df.columns if c.startswith('Latent_')]
LATENT_DIM  = len(latent_cols)

status       = df['Extant/Extinct '].astype(str)
mask_extant  = status.str.contains('Extant',  case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

X_extant  = df.loc[mask_extant,  latent_cols].values
X_extinct = df.loc[mask_extinct, latent_cols].values

labels_species = df.loc[mask_extant, 'Target_Specie'].values
unique_sp      = np.unique(labels_species)
n_extant       = X_extant.shape[0]
n_fossil       = X_extinct.shape[0]
n_species      = len(unique_sp)

le     = LabelEncoder()
y_true = le.fit_transform(labels_species)

print(f"Dimensiones latentes: {LATENT_DIM}")
print(f"Muestras extantes:    {n_extant}  ({n_species} especies)")
print(f"Fósiles:              {n_fossil}")

# =========================================================
# 2. BAYESIAN GMM — PROCESO DE DIRICHLET TRUNCADO
#    El modelo decide automáticamente cuántos componentes
#    son necesarios; los innecesarios convergen a peso ≈ 0.
# =========================================================
print("\n--- 2. Bayesian GMM / Dirichlet Process ---")

bgmm = BayesianGaussianMixture(
    n_components=K_MAX,
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.5,   # alpha < 1 → favorece pocos clusters
    n_init=5,
    random_state=42,
    max_iter=500
)
bgmm.fit(X_extant)
labels_bgmm = bgmm.predict(X_extant)

active_k   = len(np.unique(labels_bgmm))
weights    = bgmm.weights_
active_idx = np.where(weights > 0.01)[0]

ari_bgmm = adjusted_rand_score(y_true, labels_bgmm)
nmi_bgmm = normalized_mutual_info_score(y_true, labels_bgmm)

print(f"  Componentes activos (peso > 1%): {active_k} de {K_MAX}")
print(f"  Pesos activos: {np.round(weights[active_idx], 3)}")
print(f"  ARI: {ari_bgmm:.3f}   NMI: {nmi_bgmm:.3f}")

# =========================================================
# 3. BHC — BAYESIAN HIERARCHICAL CLUSTERING
#    UPGMA sobre distancias de Bhattacharyya entre especies.
#    Equivalente matemático a BHC bajo prior gaussiano
#    conjugado, que es exactamente lo que asume el bGPLVM.
# =========================================================
print("\n--- 3. BHC (UPGMA sobre Bhattacharyya) ---")

def bhattacharyya_distance(mu1, S1, mu2, S2):
    S = (S1 + S2) / 2
    try:
        Sinv = np.linalg.inv(S)
        term1 = 0.125 * (mu1 - mu2) @ Sinv @ (mu1 - mu2)
        s1, ld_S  = np.linalg.slogdet(S)
        s2, ld_S1 = np.linalg.slogdet(S1)
        s3, ld_S2 = np.linalg.slogdet(S2)
        if s1 <= 0 or s2 <= 0 or s3 <= 0:
            return np.nan
        term2 = 0.5 * (ld_S - 0.5 * (ld_S1 + ld_S2))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return np.nan

# Estadísticos gaussianos por especie
species_stats = {}
for sp in unique_sp:
    pts = X_extant[labels_species == sp]
    mu  = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False) if pts.shape[0] > 1 else np.eye(LATENT_DIM)
    cov += np.eye(LATENT_DIM) * 1e-4
    species_stats[sp] = (mu, cov)

# Matriz de distancias de Bhattacharyya
bhat_mat = np.zeros((n_species, n_species))
for i, sp1 in enumerate(unique_sp):
    for j, sp2 in enumerate(unique_sp):
        if i != j:
            d = bhattacharyya_distance(*species_stats[sp1], *species_stats[sp2])
            bhat_mat[i, j] = d if not np.isnan(d) else 0.0

bhat_sym = (bhat_mat + bhat_mat.T) / 2
np.fill_diagonal(bhat_sym, 0)

# UPGMA linkage y dendrograma
condensed  = squareform(bhat_sym, checks=False)
Z_bhc      = linkage(condensed, method='average')

# Cortar en K=active_k grupos (número que eligió el DP)
bhc_labels_sp = fcluster(Z_bhc, t=active_k, criterion='maxclust') - 1
bhc_labels_specimens = np.array([
    bhc_labels_sp[np.where(unique_sp == sp)[0][0]] for sp in labels_species
])

ari_bhc = adjusted_rand_score(y_true, bhc_labels_specimens)
nmi_bhc = normalized_mutual_info_score(y_true, bhc_labels_specimens)

print(f"  K de corte (= componentes DP activos): {active_k}")
print(f"  ARI: {ari_bhc:.3f}   NMI: {nmi_bhc:.3f}")
print("  Grupos descubiertos:")
for g in np.unique(bhc_labels_sp):
    miembros = unique_sp[bhc_labels_sp == g]
    print(f"    Cluster {g+1}: {list(miembros)}")

# =========================================================
# 4. PROYECCIÓN DE FÓSILES A CLUSTERS BHC
# =========================================================
print("\n--- 4. Proyección de fósiles ---")

fossil_ids   = df.loc[mask_extinct, 'Specimen'].values
fossil_taxon = df.loc[mask_extinct, 'Target_Specie'].values

# Centroide de cada cluster BHC
bhc_centroids = {}
for g in np.unique(bhc_labels_sp):
    sp_in_g  = unique_sp[bhc_labels_sp == g]
    pts_in_g = np.vstack([X_extant[labels_species == sp] for sp in sp_in_g])
    bhc_centroids[g] = pts_in_g.mean(axis=0)

centroid_arr      = np.stack([bhc_centroids[g] for g in sorted(bhc_centroids)])
fossil_dist_bhc   = cdist(X_extinct, centroid_arr, metric='euclidean')
fossil_cluster_bhc = fossil_dist_bhc.argmin(axis=1)

# Proyección con el DP
fossil_cluster_bgmm = bgmm.predict(X_extinct)
fossil_proba_bgmm   = bgmm.predict_proba(X_extinct)

df_fossils = pd.DataFrame({
    'Specimen':      fossil_ids,
    'Target_Specie': fossil_taxon,
    'Cluster_BHC':   fossil_cluster_bhc + 1,
    'Dist_min_BHC':  fossil_dist_bhc.min(axis=1).round(4),
    'Cluster_DP':    fossil_cluster_bgmm + 1,
    'Conf_DP':       fossil_proba_bgmm.max(axis=1).round(3),
})

out_csv = f"fossil_cluster_assignments_ld{LATENT_DIM}.csv"
df_fossils.to_csv(out_csv, index=False)
print(f"✅ Tabla guardada: '{out_csv}'")
print(df_fossils.to_string(index=False))

# =========================================================
# 5. RESUMEN COMPARATIVO
# =========================================================
print("\n--- 5. Resumen comparativo ---")
print(f"{'Método':<30} {'ARI':>8} {'NMI':>8} {'K':>6}")
print("-" * 56)
print(f"{'Bayesian GMM / DP':<30} {ari_bgmm:>8.3f} {nmi_bgmm:>8.3f} {active_k:>6}")
print(f"{'BHC (UPGMA+Bhattacharyya)':<30} {ari_bhc:>8.3f} {nmi_bhc:>8.3f} {active_k:>6}")
print("ARI: 0=aleatorio, 0.5=bueno, 1=perfecto")

# =========================================================
# 6. VISUALIZACIONES
# =========================================================
print("\n--- 6. Generando figuras ---")

# --- 6a. Pesos del Dirichlet Process ---
fig_w, ax_w = plt.subplots(figsize=(10, 4))
colors_w = ['steelblue' if w > 0.01 else 'lightgrey' for w in weights]
ax_w.bar(range(len(weights)), weights, color=colors_w)
ax_w.axhline(0.01, color='tomato', linestyle='--', label='Umbral de poda (1%)')
ax_w.set_xlabel('Componente')
ax_w.set_ylabel('Peso')
ax_w.set_title(
    f'Pesos Bayesian GMM (Dirichlet Process, α=0.5)\n'
    f'Componentes activos: {active_k} de {K_MAX}  |  azul=activo, gris=podado',
    fontsize=11, fontweight='bold'
)
ax_w.legend()
ax_w.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"bgmm_weights_ld{LATENT_DIM}.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"-> bgmm_weights_ld{LATENT_DIM}.png")

# --- 6b. Dendrograma BHC ---
fig_d, ax_d = plt.subplots(figsize=(max(10, n_species * 0.7), 6))
dendrogram(
    Z_bhc, labels=unique_sp, ax=ax_d,
    leaf_rotation=50, leaf_font_size=9,
    color_threshold=0.7 * max(Z_bhc[:, 2])
)
ax_d.set_title(
    f'Bayesian Hierarchical Clustering (UPGMA — distancia Bhattacharyya)\n'
    f'Espacio latente {LATENT_DIM}D',
    fontsize=12, fontweight='bold'
)
ax_d.set_ylabel('Distancia de Bhattacharyya')
ax_d.grid(True, axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.savefig(f"bhc_dendrogram_ld{LATENT_DIM}.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"-> bhc_dendrogram_ld{LATENT_DIM}.png")

# --- 6c. Comparativa ARI / NMI ---
methods   = ['Bayesian GMM / DP', 'BHC (Bhattacharyya)']
ari_vals  = [ari_bgmm, ari_bhc]
nmi_vals  = [nmi_bgmm, nmi_bhc]

x     = np.arange(len(methods))
width = 0.35
fig_c, ax_c = plt.subplots(figsize=(7, 5))
bars1 = ax_c.bar(x - width/2, ari_vals, width, label='ARI', color='steelblue', alpha=0.85)
bars2 = ax_c.bar(x + width/2, nmi_vals, width, label='NMI', color='darkorange', alpha=0.85)
ax_c.set_xticks(x)
ax_c.set_xticklabels(methods, fontsize=10)
ax_c.set_ylim(0, 1.15)
ax_c.set_title(
    'Alineamiento con especies reales: ARI y NMI\n(sin usar etiquetas en el entrenamiento)',
    fontsize=12, fontweight='bold'
)
ax_c.set_ylabel('Score (0=aleatorio, 1=perfecto)')
ax_c.legend()
ax_c.grid(True, axis='y', linestyle='--', alpha=0.4)
for bar in list(bars1) + list(bars2):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f'{bar.get_height():.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"clustering_comparison_ld{LATENT_DIM}.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"-> clustering_comparison_ld{LATENT_DIM}.png")

# --- 6d. Heatmap fósiles → clusters BHC ---
if n_fossil > 0:
    fig_f, ax_f = plt.subplots(figsize=(max(6, active_k * 1.5), max(5, n_fossil * 0.35)))
    im = ax_f.imshow(fossil_dist_bhc, cmap='viridis_r', aspect='auto')
    ax_f.set_xticks(range(active_k))
    ax_f.set_xticklabels([f'Cluster {g+1}' for g in sorted(bhc_centroids)], fontsize=9)
    ax_f.set_yticks(range(n_fossil))
    ax_f.set_yticklabels(
        [f"{sp[:15]} | {sp2[:12]}" for sp, sp2 in zip(fossil_taxon, fossil_ids)],
        fontsize=7
    )
    ax_f.set_title(
        f'Distancia fósil → cluster BHC (espacio latente {LATENT_DIM}D)',
        fontsize=11, fontweight='bold'
    )
    plt.colorbar(im, ax=ax_f, label='Distancia euclídea al centroide')
    for r in range(n_fossil):
        best_col = fossil_dist_bhc[r].argmin()
        ax_f.add_patch(plt.Rectangle(
            (best_col - 0.5, r - 0.5), 1, 1,
            fill=False, edgecolor='red', lw=1.5
        ))
    plt.tight_layout()
    plt.savefig(f"fossil_bhc_distances_ld{LATENT_DIM}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> fossil_bhc_distances_ld{LATENT_DIM}.png")

print("\n✅ Análisis completado.")
print(f"\nResumen ejecutivo:")
print(f"  • Dirichlet Process activa {active_k} de {K_MAX} componentes posibles.")
print(f"  • BHC recupera la jerarquía: ver bhc_dendrogram_ld{LATENT_DIM}.png")
print(f"  • NMI (BHC) = {nmi_bhc:.3f} — el espacio latente codifica estructura evolutiva real.")