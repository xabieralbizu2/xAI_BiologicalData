"""
Selección inteligente de variables para clasificación Extant vs Extinct.

Pipeline en tres bloques, pensado para comparar y justificar la elección en el TFG:

  PASO 0  Filtro de correlación (no supervisado) .................. 107 -> ~75
  PASO 1  Horseshoe regularizado bayesiano (selección embebida) ... shrinkage
  PASO 2  Selección proyectiva aproximada (forward por LOO) ....... subconjunto mínimo

Requisitos:  pip install pymc arviz scikit-learn pandas numpy
Si quieres projpred "de verdad", instala kulprit (port en Python) o usa R::projpred
sobre el modelo de referencia horseshoe que se ajusta aquí.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SRC = 'src/dataset_nbool_std.csv'
META = ['Target_Specie', 'Extant/Extinct ', 'Specimen', 'ID', 'Target_group']
CORR_THR = 0.95
DRAWS, TUNE, CHAINS = 1500, 1500, 4
TARGET_ACCEPT = 0.99          # sube a 0.99: elimina las pocas divergencias del slab
P0 = 20                       # nº esperado de variables relevantes a priori.
                              # Más alto = global shrinkage menos agresivo, deja
                              # que más coeficientes se "enciendan" (antes 8).

# ----------------------------------------------------------------------
# CARGA
# ----------------------------------------------------------------------
df = pd.read_csv(SRC)
y = (df['Extant/Extinct '].str.strip() == 'Extinct').astype(int).values   # 1 = Extinct
X_df = df.drop(columns=[c for c in META if c in df.columns]).select_dtypes('number')
feat = np.array(X_df.columns)
Xs = StandardScaler().fit_transform(X_df.values)
print(f'Datos: {Xs.shape[0]} muestras, {Xs.shape[1]} features, Extinct={y.sum()}')

# ----------------------------------------------------------------------
# PASO 0 — FILTRO POR CORRELACIÓN (quita redundancia ISO/SSFA)
# Se queda con la feature de mayor varianza de cada grupo correlacionado.
# ----------------------------------------------------------------------
def correlation_filter(X, names, thr=0.95):
    C = np.abs(np.corrcoef(X, rowvar=False))
    var = X.var(axis=0)
    taken = np.zeros(len(names), bool)
    keep = []
    for i in np.argsort(-var):
        if taken[i]:
            continue
        keep.append(i)
        taken[(C[i] > thr)] = True
    return sorted(keep)

keep = correlation_filter(Xs, feat, CORR_THR)
Xf, feat_f = Xs[:, keep], feat[keep]
print(f'Paso 0: {len(feat)} -> {len(feat_f)} features tras filtro corr>{CORR_THR}')

# ----------------------------------------------------------------------
# PASO 1 — HORSESHOE REGULARIZADO (Piironen & Vehtari, 2017)
# Encoge a 0 los coeficientes irrelevantes. La clave frente al horseshoe
# básico es el "slab" (c2): regulariza los coeficientes grandes y estabiliza
# el sampler en régimen p≈n.
# ----------------------------------------------------------------------
n, p = Xf.shape
p0 = P0                                  # nº esperado de variables relevantes (prior)
# Desbalanceo: el log-odds base de la clase Extinct es log(35/125)≈-1.27.
# Centrar el prior del intercepto ahí evita que alpha "absorba" la señal
# intentando explicar el desbalanceo, dejando que los beta capten estructura real.
logodds_base = np.log(y.mean() / (1.0 - y.mean()))
with pm.Model() as model:
    Xd = pm.Data('X', Xf)
    # global shrinkage: escala según p0 esperado (recomendación del paper)
    tau0 = (p0 / (p - p0)) * (1.0 / np.sqrt(n))
    tau = pm.HalfCauchy('tau', beta=tau0)
    lam = pm.HalfCauchy('lam', beta=1.0, shape=p)
    # slab ablandado: media de c2 mayor => menos regularización de coeficientes
    # grandes, así una señal conjunta moderada puede emerger (antes beta=8).
    c2 = pm.InverseGamma('c2', alpha=2.0, beta=20.0)
    lam_tilde = pm.Deterministic(
        'lam_tilde', pm.math.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2)))
    z = pm.Normal('z', 0.0, 1.0, shape=p)
    beta = pm.Deterministic('beta', z * lam_tilde * tau)
    alpha = pm.Normal('alpha', logodds_base, 1.5)   # centrado en el desbalanceo
    logit = alpha + pm.math.dot(Xd, beta)
    pm.Bernoulli('y', logit_p=logit, observed=y)

    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                      target_accept=TARGET_ACCEPT, random_seed=0)

# diagnóstico de selección: |beta| posterior y prob. de que esté "encendida"
post = idata.posterior['beta'].stack(s=('chain', 'draw')).values  # (p, S)
summary = pd.DataFrame({
    'feature': feat_f,
    'mean_abs_beta': np.abs(post).mean(axis=1),
    'p_beta_pos': (post > 0).mean(axis=1),
    'hdi95_excl_0': [(np.percentile(post[i], 2.5) > 0) or
                     (np.percentile(post[i], 97.5) < 0) for i in range(p)],
    'hdi90_excl_0': [(np.percentile(post[i], 5.0) > 0) or
                     (np.percentile(post[i], 95.0) < 0) for i in range(p)],
}).sort_values('mean_abs_beta', ascending=False).reset_index(drop=True)
# "fuerza" de la evidencia de signo: cuán lejos de 0.5 está p_beta_pos
summary['evidencia_signo'] = (summary['p_beta_pos'] - 0.5).abs() * 2
print('\nVariables ordenadas por relevancia (horseshoe):')
print(summary.head(20).to_string(index=False))
summary.to_csv('seleccion_horseshoe_extant.csv', index=False)

# ----------------------------------------------------------------------
# CRITERIO DE SELECCIÓN ESCALONADO (de estricto a permisivo):
#   1) HDI 95% excluye 0   -> evidencia fuerte
#   2) HDI 90% excluye 0   -> evidencia moderada
#   3) top-N por |beta|    -> fallback si la señal es difusa
# Se usa el primero que devuelva >= 2 variables.
# ----------------------------------------------------------------------
TOP_N_FALLBACK = 12
sel95 = summary[summary['hdi95_excl_0']]['feature'].tolist()
sel90 = summary[summary['hdi90_excl_0']]['feature'].tolist()
print(f'\nHDI 95% excluye 0: {len(sel95)} variables')
print(f'HDI 90% excluye 0: {len(sel90)} variables')

if len(sel95) >= 2:
    selected, criterio = sel95, 'HDI 95%'
elif len(sel90) >= 2:
    selected, criterio = sel90, 'HDI 90%'
else:
    selected = summary['feature'].head(TOP_N_FALLBACK).tolist()
    criterio = f'top-{TOP_N_FALLBACK} por |beta| (señal difusa)'
print(f'Criterio aplicado: {criterio} -> {len(selected)} variables')
print('Variables finales:', selected)

# ----------------------------------------------------------------------
# EXPORTA EL DATASET REDUCIDO listo para entrenar
# Mantiene los metadatos originales + solo las variables seleccionadas.
# ----------------------------------------------------------------------
meta_present = [c for c in META if c in df.columns]
df_reducido = df[meta_present + selected].copy()
OUT = 'dataset_seleccionado_extant.csv'
df_reducido.to_csv(OUT, index=False)
print(f'\nDataset reducido guardado en {OUT} -> shape {df_reducido.shape}')
print(f'   ({len(meta_present)} metadatos + {len(selected)} variables seleccionadas)')

# ----------------------------------------------------------------------
# PASO 2 — ¿cuánto se pierde al reducir? Ajusta el modelo REDUCIDO (solo las
# variables seleccionadas) y el COMPLETO (las 74 tras filtro de correlación)
# y los compara con az.compare (elpd_loo). Es el argumento numérico del TFG:
# si el reducido no pierde elpd significativo, las variables descartadas
# no aportaban poder predictivo.
# ----------------------------------------------------------------------
def fit_simple(cols, coef_scale=1.0):
    with pm.Model():
        Xd = pm.Data('X', Xf[:, cols])
        b = pm.Normal('b', 0, coef_scale, shape=len(cols))
        a = pm.Normal('a', logodds_base, 1.5)
        pm.Bernoulli('y', logit_p=a + pm.math.dot(Xd, b), observed=y)
        idt = pm.sample(1000, tune=1000, chains=CHAINS,
                        target_accept=0.95, random_seed=0,
                        idata_kwargs={'log_likelihood': True})
    return idt

if selected:
    cols_sel = [list(feat_f).index(f) for f in selected]
    idata_red = fit_simple(cols_sel)
    print(f'\nLOO modelo REDUCIDO ({len(selected)} vars):')
    print(az.loo(idata_red))

    # modelo completo: 74 variables -> prior algo más estrecho para estabilizar
    idata_full = fit_simple(list(range(p)), coef_scale=0.5)
    print(f'\nLOO modelo COMPLETO ({p} vars):')
    print(az.loo(idata_full))

    # comparación formal
    cmp = az.compare({'reducido': idata_red, 'completo': idata_full},
                     ic='loo', scale='log')
    print('\n=== COMPARACIÓN (az.compare, elpd_loo) ===')
    print(cmp.to_string())
    cmp.to_csv('comparacion_loo_reducido_vs_completo.csv')

    # lectura automática para la memoria
    mejor = cmp.index[0]
    delta = cmp.iloc[1]['elpd_diff']
    se = cmp.iloc[1]['dse']
    print(f'\nMejor modelo por elpd_loo: "{mejor}".')
    print(f'Diferencia elpd = {delta:.2f} ± {se:.2f} (dse).')
    if abs(delta) < 2 * se:
        print('-> La diferencia NO es significativa (|delta| < 2*dse): el modelo')
        print('   reducido predice igual de bien con muchas menos variables. '
              'Reducción justificada.')
    else:
        print('-> La diferencia ES significativa: revisa si faltan variables '
              'relevantes en la selección.')
    print('\nEntrena tu modelo final con dataset_seleccionado_extant.csv')