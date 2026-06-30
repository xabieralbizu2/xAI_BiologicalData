import os
import json
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score,
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN
# =========================================================
# Prior del modelo bayesiano: "lasso", "gaussian", "horseshoe", "spike_and_slab"
TIPO_PRIOR = "horseshoe"
CHAINS = 4
DRAWS = 1000
TUNE = 1000              # 1000 basta para un modelo lineal pequeño (antes 1500)
TARGET_ACCEPT = 0.9      # 0.9 en vez de 0.99: NUTS da pasos razonables sin
                         # disparar el coste x5-x10. Si aparecen divergencias,
                         # la solución es reparametrizar, no volver a 0.99.

# --- Selección de variables (dentro de cada fold, SIN fuga) ---
# K_FEATURES y el resto de hiperparámetros se FIJAN por criterio previo, no se
# eligen maximizando el LOOCV (eso sería optimizar contra el test). El barrido
# exploratorio sugirió k≈30 como zona estable; se fija aquí de forma honesta.
USAR_SELECCION = True
K_FEATURES = 30

# --- Representación de entrada ---
# Combinar variables crudas (microtextura ISO/SSFA) + dimensiones latentes del
# GPLVM mejora el accuracy honesto respecto a usar cada una por separado, porque
# capturan aspectos complementarios de la señal morfológica.
USAR_LATENTES_GPLVM = True

# LOOCV exacto: mismos ajustes que el modelo final (máximo rigor).
DRAWS_CV = DRAWS
TUNE_CV = TUNE
CHECKPOINT_LOO = f"loo_checkpoint_blr_{TIPO_PRIOR}.json"

print(f"--- BLR | PRIOR: {TIPO_PRIOR.upper()} | k={K_FEATURES if USAR_SELECCION else 'todas'} "
      f"| latentes={USAR_LATENTES_GPLVM} ---")

# =========================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================================================
print("Cargando y filtrando datos...")
df = pd.read_csv('src/dataset_seleccionado_extant.csv')
df['Extant/Extinct '] = df['Extant/Extinct '].str.strip()
extant_df = df[df['Extant/Extinct '] == 'Extant'].copy().reset_index(drop=True)

X_raw = extant_df.drop(columns=['Target_Specie', 'Extant/Extinct ', 'Specimen', 'ID'],
                       errors='ignore')

# Añadir dimensiones latentes del GPLVM si se solicita
if USAR_LATENTES_GPLVM and os.path.exists('src/espacio_latente_gplvm.csv'):
    lat = pd.read_csv('src/espacio_latente_gplvm.csv')
    lat['Extant/Extinct '] = lat['Extant/Extinct '].str.strip()
    lat_ext = lat[lat['Extant/Extinct '] == 'Extant'].copy().reset_index(drop=True)
    lat_cols = [c for c in lat_ext.columns if c.startswith('Latent_')]
    # Alinear por orden de fila (ambos provienen del mismo subconjunto extant)
    if len(lat_ext) == len(X_raw):
        X_raw = pd.concat([X_raw.reset_index(drop=True),
                           lat_ext[lat_cols].reset_index(drop=True)], axis=1)
        print(f"  Añadidas {len(lat_cols)} dimensiones latentes del GPLVM.")
    else:
        print("  AVISO: nº de filas extant no coincide entre datasets; "
              "se omiten las latentes.")

y_raw = extant_df['Target_Specie']
feature_names = X_raw.columns.tolist()

le = LabelEncoder()
y_idx = le.fit_transform(y_raw)
n_classes = len(le.classes_)
X_full = X_raw.values
n_features_total = X_full.shape[1]

print(f"Especies a clasificar : {n_classes}")
print(f"Variables disponibles : {n_features_total}")
print(f"Muestras (Extant)     : {len(y_idx)}")


# =========================================================
# 2. SELECCIÓN DE VARIABLES (SIN FUGA)
#    Ajusta scaler + SelectKBest SOLO con el train que se le
#    pase. Devuelve los arrays transformados y los objetos
#    ajustados para aplicar la MISMA transformación al test.
# =========================================================
def ajustar_preproceso(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    selector = None
    if USAR_SELECCION:
        k = min(K_FEATURES, X_tr_s.shape[1])
        selector = SelectKBest(f_classif, k=k).fit(X_tr_s, y_tr)
        X_tr_s = selector.transform(X_tr_s)
    return X_tr_s, scaler, selector


def aplicar_preproceso(X, scaler, selector):
    X_s = scaler.transform(X)
    if selector is not None:
        X_s = selector.transform(X_s)
    return X_s


# =========================================================
# 3. MOTOR BAYESIANO (FUNCIÓN REUTILIZABLE)
# =========================================================
def construir_y_muestrear_modelo(X_val, y_val, n_classes, tipo_prior,
                                 draws=DRAWS, tune=TUNE, progressbar=True):
    n_features = X_val.shape[1]
    with pm.Model() as model:
        X_data = pm.Data('X', X_val)
        y_data = pm.Data('y', y_val)

        alpha_rest = pm.Normal('alpha_rest', mu=0, sigma=5, shape=n_classes - 1)
        shape_beta = (n_features, n_classes - 1)

        if tipo_prior == "lasso":
            beta_rest = pm.Laplace('beta_rest', mu=0, b=0.1, shape=shape_beta)
        elif tipo_prior == "gaussian":
            beta_rest = pm.Normal('beta_rest', mu=0, sigma=1.0, shape=shape_beta)
        elif tipo_prior == "horseshoe":
            tau = pm.HalfCauchy('tau', beta=1)
            lam = pm.HalfCauchy('lam', beta=1, shape=shape_beta)
            z = pm.Normal('z', mu=0, sigma=1, shape=shape_beta)
            beta_rest = pm.Deterministic('beta_rest', z * tau * lam)
        elif tipo_prior == "spike_and_slab":
            w_local = pm.Beta('w_local', alpha=1, beta=1, shape=shape_beta)
            spike = pm.Normal('spike', mu=0, sigma=0.05, shape=shape_beta)
            slab = pm.Normal('slab', mu=0, sigma=5.0, shape=shape_beta)
            beta_rest = pm.Deterministic('beta_rest', w_local * slab + (1 - w_local) * spike)
        else:
            raise ValueError("TIPO_PRIOR no reconocido.")

        alpha = pt.concatenate([pt.zeros(1), alpha_rest])
        beta = pt.concatenate([pt.zeros((n_features, 1)), beta_rest], axis=1)

        mu = pm.math.dot(X_data, beta) + alpha
        p = pm.Deterministic('p', pm.math.softmax(mu, axis=1))
        y_obs = pm.Categorical('y_obs', p=p, observed=y_data)

        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=42,
                          progressbar=progressbar)
    return model, idata


def predecir_proba_bayesiano(model, idata, X_new):
    with model:
        pm.set_data({'X': X_new, 'y': np.zeros(len(X_new), dtype=int)})
        ppc = pm.sample_posterior_predictive(
            idata, predictions=True, var_names=['p'],
            random_seed=42, progressbar=False)
    p_samples = ppc.predictions['p'].values
    p_mean = p_samples.reshape(-1, len(X_new), p_samples.shape[-1]).mean(axis=0)
    return p_mean


# =========================================================
# 4. MODELO FINAL (todos los extant, con selección global)
#    Solo para distribuciones posteriori / diagnóstico / fósiles.
#    OJO: este preproceso global usa todos los datos; es válido
#    para INTERPRETAR y para PREDECIR FÓSILES, pero NO se usa
#    para estimar accuracy (eso lo hace el LOOCV de la sección 6).
# =========================================================
print(f"\nAjustando preproceso global y muestreando modelo final ({TIPO_PRIOR})...")
X_final, scaler_global, selector_global = ajustar_preproceso(X_full, y_idx)
sel_names = (np.array(feature_names)[selector_global.get_support()].tolist()
             if selector_global is not None else feature_names)
n_features = X_final.shape[1]
print(f"  Variables seleccionadas ({n_features}): {sel_names}")

bayesian_logistic, idata = construir_y_muestrear_modelo(
    X_final, y_idx, n_classes, TIPO_PRIOR, draws=DRAWS, tune=TUNE, progressbar=True)

with bayesian_logistic:
    pm.compute_log_likelihood(idata)

# --- Distribución posterior de los coeficientes (beta_rest) ---
# beta_rest tiene forma (n_features, n_classes-1). Etiquetamos cada coordenada
# con el nombre de la variable seleccionada y la clase a la que corresponde
# (la clase 0 es la referencia y queda fijada a 0, por eso son n_classes-1).
print("\nGuardando distribución posterior de los coeficientes (beta_rest)...")
coords_beta = {
    "beta_rest_dim_0": sel_names,
    "beta_rest_dim_1": list(le.classes_[1:]),
}
try:
    idata = idata.assign_coords(coords_beta)
except Exception as e:
    print(f"  AVISO: no se pudieron asignar coords descriptivas ({e}).")

# --- Figura ÚNICA y legible: top coeficientes (variable x clase) ---
# Graficar las 30 variables x (n_classes-1) clases = ~180 coeficientes es
# ilegible y, además, casi todos están contraídos a cero. En lugar de
# seleccionar VARIABLES (que arrastran sus clases nulas), seleccionamos los
# COEFICIENTES INDIVIDUALES que más se alejan del cero y los mostramos en un
# único forest plot ordenado por magnitud. Las dimensiones latentes del GPLVM
# se excluyen por no ser interpretables como descriptores de microtextura.
TOP_COEFS = 10

post = idata.posterior['beta_rest']                   # (chain, draw, dim0, dim1)
clases = list(le.classes_[1:])
# Magnitud posterior |media| de cada coeficiente (variable, clase).
mean_abs = np.abs(post.mean(dim=['chain', 'draw']).values)   # (n_features, n_clases)

# Construir lista de coeficientes candidatos, excluyendo latentes del GPLVM.
candidatos = []
for vi, vname in enumerate(sel_names):
    if str(vname).startswith('Latent_'):
        continue
    for ci, cname in enumerate(clases):
        candidatos.append((mean_abs[vi, ci], vname, cname))
candidatos.sort(reverse=True)
sel = candidatos[:min(TOP_COEFS, len(candidatos))]
print(f"  Mostrando {len(sel)} coeficientes con mayor |media| posterior.")

# HDI 94% por muestreo directo.
def hdi(x, p=0.94):
    xs = np.sort(x)
    n = len(xs); k = int(np.floor(p * n))
    widths = xs[k:] - xs[:n - k]
    j = int(np.argmin(widths))
    return xs[j], xs[j + k]

filas = []
for _, vname, cname in sel:
    vi = sel_names.index(vname); ci = clases.index(cname)
    draws = post.values[:, :, vi, ci].ravel()
    lo, hi = hdi(draws)
    filas.append((f"{vname}  →  {cname}", float(np.median(draws)), lo, hi))

# Ordenar por magnitud (menor arriba; al ser eje y creciente, el mayor queda abajo).
filas.sort(key=lambda r: abs(r[1]))
fig_h = max(2.6, 0.34 * len(filas) + 0.9)
fig, ax = plt.subplots(figsize=(7, fig_h))
ypos = np.arange(len(filas))
for y, (lab, med, lo, hi) in zip(ypos, filas):
    color = '#1f77b4' if (lo > 0 or hi < 0) else '#9aa0a6'  # azul si HDI excluye 0
    ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle='round')
    ax.plot(med, y, 'o', color=color, ms=5.5, mfc='white', mew=1.5)
ax.axvline(0, color='red', ls='--', lw=1.1)
ax.set_yticks(ypos)
ax.set_yticklabels([f[0] for f in filas], fontsize=8.5)
ax.set_xlabel('Coeficiente β (HDI 94%)', fontsize=9.5)
ax.set_title(f'Coeficientes posteriores con mayor magnitud — BLR ({TIPO_PRIOR.upper()})',
             fontsize=10.5, fontweight='bold')
ax.margins(y=0.03)
ax.grid(axis='x', ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'5_posterior_coeficientes_{TIPO_PRIOR}.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"-> Guardado: '5_posterior_coeficientes_{TIPO_PRIOR}.png'")

# --- Figura 6: densidades posteriores de ESOS MISMOS coeficientes top ---
# Rejilla compacta (máx. 5 columnas) con una densidad por coeficiente top,
# línea de referencia en cero. Complementa al forest plot mostrando la FORMA
# de cada posterior (pico en cero + cola lateral en los pocos con señal).
sel_orden = sorted(sel, key=lambda r: -r[0])   # mayor magnitud primero
n_panels = len(sel_orden)
ncol = min(5, n_panels)
nrow = int(np.ceil(n_panels / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.7 * nrow),
                         squeeze=False)
for k, (_, vname, cname) in enumerate(sel_orden):
    r, c = divmod(k, ncol)
    axk = axes[r][c]
    vi = sel_names.index(vname); ci = clases.index(cname)
    draws = post.values[:, :, vi, ci].ravel()
    lo, hi = hdi(draws)
    axk.hist(draws, bins=60, density=True, color='#1f77b4', alpha=0.55)
    axk.axvline(0, color='red', ls='--', lw=1.0)
    axk.plot([lo, hi], [0, 0], color='k', lw=3, solid_capstyle='butt')
    axk.set_title(f"{vname}\n→ {cname}", fontsize=7.2)
    axk.set_yticks([]); axk.tick_params(axis='x', labelsize=6.5)
# apagar ejes sobrantes
for k in range(n_panels, nrow * ncol):
    r, c = divmod(k, ncol); axes[r][c].axis('off')
fig.suptitle(f'Densidades posteriores de los coeficientes top — BLR '
             f'({TIPO_PRIOR.upper()})', fontsize=10.5, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'6_posterior_densidades_{TIPO_PRIOR}.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"-> Guardado: '6_posterior_densidades_{TIPO_PRIOR}.png'")

# =========================================================
# 5. DIAGNÓSTICO PSIS-LOO (complementario, NO es accuracy)
# =========================================================
print("\nDiagnóstico PSIS-LOO (ELPD y Pareto k-hat)...")
loo_result = az.loo(idata, pointwise=True)
loo_text = str(loo_result)
print(loo_result)

plt.figure(figsize=(8, 5))
az.plot_khat(loo_result, show_bins=True)
plt.title(f"Diagnóstico de Pareto k - {TIPO_PRIOR.upper()}")
plt.axhline(0.7, color='red', linestyle='--', label='Límite (0.7)')
plt.legend(); plt.tight_layout()
plt.savefig(f'2_loo_khat_plot_{TIPO_PRIOR}.png', dpi=300)
print(f"-> Guardado: '2_loo_khat_plot_{TIPO_PRIOR}.png'")

# =========================================================
# 6. LOOCV EXACTO — accuracy honesta (selección DENTRO del fold)
# =========================================================
print("\n" + "=" * 55)
print(" LOOCV EXACTO (BLR) — reentrena el modelo N veces")
print(" Selección de variables ajustada DENTRO de cada fold")
print("=" * 55)

loo = LeaveOneOut()
folds = list(loo.split(X_full))
n = len(y_idx)

resultados = {}
if os.path.exists(CHECKPOINT_LOO):
    with open(CHECKPOINT_LOO) as f:
        resultados = {int(k): v for k, v in json.load(f).items()}
    print(f"  Checkpoint: {len(resultados)}/{n} folds ya resueltos.")

folds_pendientes = [(fold, te) for fold, (tr, te) in enumerate(folds, start=1)
                    if fold not in resultados]

for fold, te_idx in tqdm(folds_pendientes, total=len(folds_pendientes),
                         desc="Progreso LOOCV", unit="fold"):
    i = te_idx[0]
    tr_idx = np.setdiff1d(np.arange(n), te_idx)

    # --- Preproceso + selección ajustados SOLO con el train del fold ---
    X_tr_s, sc_f, sel_f = ajustar_preproceso(X_full[tr_idx], y_idx[tr_idx])
    X_te_s = aplicar_preproceso(X_full[te_idx], sc_f, sel_f)

    model_fold, idata_fold = construir_y_muestrear_modelo(
        X_tr_s, y_idx[tr_idx], n_classes, TIPO_PRIOR,
        draws=DRAWS_CV, tune=TUNE_CV, progressbar=False)

    p_te = predecir_proba_bayesiano(model_fold, idata_fold, X_te_s)
    pred = int(np.argmax(p_te, axis=1)[0])

    resultados[fold] = {'idx': int(i), 'y_true': int(y_idx[i]), 'y_pred': pred}
    with open(CHECKPOINT_LOO, "w") as f:
        json.dump({str(k): v for k, v in resultados.items()}, f, indent=2)

y_true_loo = np.array([resultados[fold]['y_true'] for fold in sorted(resultados)])
y_pred_loo = np.array([resultados[fold]['y_pred'] for fold in sorted(resultados)])

acc_loo = accuracy_score(y_true_loo, y_pred_loo)
bacc_loo = balanced_accuracy_score(y_true_loo, y_pred_loo)
f1_loo = f1_score(y_true_loo, y_pred_loo, average='macro')

print("\n--- MÉTRICAS LOOCV (nivel especie) ---")
print(f"Accuracy           : {acc_loo:.2%}")
print(f"Balanced accuracy  : {bacc_loo:.2%}")
print(f"F1 macro           : {f1_loo:.2%}")

report_loo = classification_report(
    y_true_loo, y_pred_loo, labels=np.arange(n_classes),
    target_names=le.classes_)
print("\n" + report_loo)

pd.DataFrame({
    'y_true': le.inverse_transform(y_true_loo),
    'y_pred_loo': le.inverse_transform(y_pred_loo),
}).to_csv(f"predicciones_loocv_blr_{TIPO_PRIOR}.csv", index=False)
print(f"-> Guardado: 'predicciones_loocv_blr_{TIPO_PRIOR}.csv'")

cm_loo = confusion_matrix(y_true_loo, y_pred_loo, labels=np.arange(n_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm_loo, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Matriz de Confusión LOOCV — BLR ({TIPO_PRIOR.upper()})',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicción (LOO)'); plt.ylabel('Especie Real')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(f'3_matriz_confusion_loocv_{TIPO_PRIOR}.png', dpi=300)
print(f"-> Guardado: '3_matriz_confusion_loocv_{TIPO_PRIOR}.png'")

# Resumen como imagen
texto_final = (f"=== MÉTRICAS LOOCV EXACTO ({TIPO_PRIOR.upper()}) ===\n"
               f"Representación: crudas{'+latentes' if USAR_LATENTES_GPLVM else ''} | "
               f"k={K_FEATURES if USAR_SELECCION else 'todas'}\n"
               f"Accuracy          : {acc_loo:.4f}\n"
               f"Balanced accuracy : {bacc_loo:.4f}\n"
               f"F1 macro          : {f1_loo:.4f}\n\n"
               + report_loo + "\n\n"
               f"=== DIAGNÓSTICO PSIS-LOO ({TIPO_PRIOR.upper()}) ===\n" + loo_text)
fig_text, ax_text = plt.subplots(figsize=(10, 9)); ax_text.axis('off')
ax_text.text(0.01, 0.99, texto_final, fontsize=11, fontfamily='monospace',
             va='top', ha='left', transform=ax_text.transAxes)
plt.savefig(f'4_metricas_resumen_{TIPO_PRIOR}.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print(f"-> Guardado: '4_metricas_resumen_{TIPO_PRIOR}.png'")

# =========================================================
# 7. PROYECCIÓN DE FÓSILES (usa preproceso global de sección 4)
# =========================================================
print("\n" + "=" * 55)
print(" PROYECCIÓN DE FÓSILES (EXTINCT)")
print("=" * 55)

df_full = pd.read_csv('src/dataset_nbool_std.csv')
df_full['Extant/Extinct '] = df_full['Extant/Extinct '].str.strip()
extinct_df = df_full[df_full['Extant/Extinct '] == 'Extinct'].copy()

if len(extinct_df) == 0:
    print("No se encontraron filas 'Extinct'.")
else:
    X_ext_raw = extinct_df.drop(
        columns=['Target_Specie', 'Extant/Extinct ', 'Specimen', 'ID'],
        errors='ignore')
    # Reindexar a las columnas crudas; las latentes de fósiles, si no existen,
    # se rellenan a 0 (el GPLVM no proyecta fósiles aquí). Si tienes latentes
    # de fósiles, cárgalas igual que en la sección 1.
    X_ext_raw = X_ext_raw.reindex(columns=feature_names, fill_value=0)
    X_ext_proc = aplicar_preproceso(X_ext_raw.values, scaler_global, selector_global)
    extinct_ids = extinct_df['Specimen'].values

    print(f"Muestras extinct a predecir: {len(extinct_ids)}")
    p_mean_ext = predecir_proba_bayesiano(bayesian_logistic, idata, X_ext_proc)
    best_idx_ext = np.argmax(p_mean_ext, axis=1)
    best_sp_ext = le.inverse_transform(best_idx_ext)
    best_prob_ext = np.max(p_mean_ext, axis=1)

    rows = []
    for i, specimen in enumerate(extinct_ids):
        row = {'Specimen_Extinct': specimen}
        for c, sp in enumerate(le.classes_):
            row[f"P_{sp.replace(' ', '_')}"] = np.round(p_mean_ext[i, c], 4)
        row['Best_Extant_Specie'] = best_sp_ext[i]
        row['Best_Extant_Prob'] = np.round(best_prob_ext[i], 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"predicciones_extinct_blr_{TIPO_PRIOR}.csv", index=False)
    print(f"-> Guardado: 'predicciones_extinct_blr_{TIPO_PRIOR}.csv'")

print("\n✅ Ejecución completada: LOOCV honesto + PSIS-LOO + proyección de fósiles.")