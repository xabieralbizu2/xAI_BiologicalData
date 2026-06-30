import os
import json
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, accuracy_score,
    f1_score, classification_report,
)
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
TIPO_PRIOR = "horseshoe"  # Opciones: "lasso", "gaussian", "horseshoe", "spike_and_slab"
FILE_PATH = "src/dataset_con_grupos.csv"
CHAINS = 4
DRAWS = 1000
TUNE = 1000              # 1000 basta para modelos lineales pequeños (antes 1500)
TARGET_ACCEPT = 0.9      # 0.9 en vez de 0.99: evita el coste x5-x10 de NUTS

# --- Jerarquía a usar (de notes/hierarchies.txt) ---
#   "H3"     : L1 Theropithecus vs Other ; L2 Papio vs Other3 ; L3 especie (código original)
#   "H2"     : L1 {Papio+Lophocebus} vs Other vs Theropithecus ; L2 especie dentro de grupo
#   "DENDRO" : jerarquía binaria DERIVADA DE LOS DATOS. Se construye un dendrograma
#              (clustering jerárquico Ward) sobre los centroides de cada especie en el
#              espacio latente del GPLVM, y se clasifica en cascada bajando por el árbol
#              binario nodo a nodo hasta llegar a la especie (hoja). A diferencia de
#              H2/H3 (cableadas a mano), aquí la estructura la aprenden los datos.
#   "TREE"   : jerarquía binaria FIJA que defines tú a mano en ARBOL_FIJO (abajo).
#              Se entrena un clasificador binario bayesiano en CADA nodo interno del
#              árbol y se clasifica en cascada multiplicando P(rama) hasta la hoja,
#              EXACTAMENTE igual que DENDRO, pero sin recalcular el árbol nunca (no
#              usa GPLVM ni Bhattacharyya: el árbol es el que tú pasas). Para el árbol
#              de la imagen (7 especies) se entrenan 6 submodelos (6 nodos internos).
# En el barrido honesto H2 (0.272) superó a H3 (0.208) a nivel especie.
JERARQUIA = "TREE"

# --- Parámetros del dendrograma (solo se usan si JERARQUIA == "DENDRO") ---
DENDRO_LINKAGE = "ward"        # "ward" recupera el clado Papio y da un árbol binario
                               # balanceado; alternativas: "average", "complete".
DENDRO_METRIC = "euclidean"    # distancia entre centroides en el espacio latente

# --- Árbol FIJO (solo se usa si JERARQUIA == "TREE") ---
# Se especifica como LISTA ANIDADA: cada nodo interno es una lista/tupla de 2 elementos
# [izquierda, derecha]; cada hoja es el nombre EXACTO de la especie (string), tal como
# aparece en la columna 'Target_Specie'. Los paréntesis anidados reflejan directamente
# la forma del dendrograma (los de fuera son las separaciones más altas/tempranas).
#
# Este árbol reproduce el dendrograma de Bhattacharyya (UPGMA) de la figura:
#   raíz: Theropithecus gelada  |  resto
#     resto: {Papio ursinus, Lophocebus, P. anubis, P. hamadryas}  |  {Cercocebus, Macaca}
#       grupo Papio+Lopho: Papio ursinus  |  {Lophocebus, P. anubis, P. hamadryas}
#         {Lophocebus, P. anubis, P. hamadryas}: Lophocebus  |  {P. anubis, P. hamadryas}
#           {P. anubis, P. hamadryas}: Papio anubis | Papio hamadryas
#       grupo Cerco+Macaca: Cercocebus atys | Macaca sylvanus
# -> 7 hojas, 6 nodos internos = 6 submodelos binarios.
ARBOL_FIJO = [
    "Theropithecus gelada",
    [
        [
            "Papio ursinus",
            [
                "Papio anubis",
                "Papio hamadryas",
            ],
        ],
        [
            "Macaca sylvanus",
            [
                "Cercocebus atys",
                "Lophocebus aterrimus",
            ],
        ],
    ],
]

# --- Selección de variables (dentro de cada fold y por nivel, SIN fuga) ---
USAR_SELECCION = True
K_FEATURES = 30          # fijada por criterio previo, NO maximizando el LOOCV

# --- Combinar variables crudas + latentes GPLVM ---
USAR_LATENTES_GPLVM = True
GPLVM_PATH = "src/espacio_latente_gplvm.csv"

# LOOCV exacto: reentrena la jerarquía completa (4 modelos) en cada fold con los
# mismos ajustes que el modelo final (máximo rigor). PyMC ya paraleliza las
# CHAINS cadenas de cada submodelo internamente. Sobre eso se añade tqdm (barra
# de progreso) y checkpointing por fold para poder reanudar.
DRAWS_CV = DRAWS
TUNE_CV = TUNE

# CHECKPOINT_LOO : archivo donde se guarda el resultado de cada fold según se
#                  completa. Si el proceso se interrumpe, al relanzar retoma
#                  desde donde se quedó.
CHECKPOINT_LOO = f"loo_checkpoint_3niveles_{TIPO_PRIOR}.json"

print(f"--- MODELO JERÁRQUICO BAYESIANO (3 NIVELES) + LOOCV | PRIOR: {TIPO_PRIOR.upper()} ---")

# =========================================================
#  MAPEOS JERÁRQUICOS — seleccionables según JERARQUIA
# =========================================================
# --- H3 (código original): L1 Theropithecus vs Other; L2 Papio vs Other3 ---
L1_MAP_H3 = {
    "Theropithecus gelada":  "Theropithecus",
    "Papio anubis":          "Other_L1",
    "Papio hamadryas":       "Other_L1",
    "Papio ursinus":         "Other_L1",
    "Lophocebus aterrimus":  "Other_L1",
    "Cercocebus atys":       "Other_L1",
    "Macaca sylvanus":       "Other_L1",
}
L2_MAP_H3 = {
    "Papio anubis":          "Grupo_Papio",
    "Papio hamadryas":       "Grupo_Papio",
    "Papio ursinus":         "Grupo_Papio",
    "Lophocebus aterrimus":  "Grupo_Other3",
    "Cercocebus atys":       "Grupo_Other3",
    "Macaca sylvanus":       "Grupo_Other3",
}

# --- H2: L1 {Papio+Lophocebus} vs Other vs Theropithecus ---
# L1 con 3 ramas; Theropithecus es terminal (monoespecífico en extant).
L1_MAP_H2 = {
    "Theropithecus gelada":  "Theropithecus",
    "Papio anubis":          "Grupo_PL",
    "Papio hamadryas":       "Grupo_PL",
    "Papio ursinus":         "Grupo_PL",
    "Lophocebus aterrimus":  "Grupo_PL",
    "Cercocebus atys":       "Grupo_Other",
    "Macaca sylvanus":       "Grupo_Other",
}
# En H2 el nivel 2 es directamente la especie dentro de cada rama no terminal.
# Se modela como "experto de especie" por grupo, igual que en H3.
L2_MAP_H2 = {
    "Papio anubis":          "Grupo_PL",
    "Papio hamadryas":       "Grupo_PL",
    "Papio ursinus":         "Grupo_PL",
    "Lophocebus aterrimus":  "Grupo_PL",
    "Cercocebus atys":       "Grupo_Other",
    "Macaca sylvanus":       "Grupo_Other",
}

if JERARQUIA == "H3":
    L1_MAP, L2_MAP = L1_MAP_H3, L2_MAP_H3
    EXPERT_GROUPS = ["Grupo_Papio", "Grupo_Other3"]
    OTHER_L1_LABEL = "Other_L1"
elif JERARQUIA == "H2":
    L1_MAP, L2_MAP = L1_MAP_H2, L2_MAP_H2
    EXPERT_GROUPS = ["Grupo_PL", "Grupo_Other"]
    OTHER_L1_LABEL = None  # en H2 el L1 ya da el grupo directamente
elif JERARQUIA == "DENDRO":
    # La estructura se construye más abajo, tras cargar los datos y el espacio
    # latente del GPLVM (necesitamos los centroides por especie). Aquí solo se
    # marca que se usará el motor genérico de árbol binario.
    L1_MAP = L2_MAP = None
    EXPERT_GROUPS = None
    OTHER_L1_LABEL = None
elif JERARQUIA == "TREE":
    # El árbol es FIJO (ARBOL_FIJO). No hay nada que derivar de los datos ni del
    # GPLVM: se usa el mismo motor genérico de árbol binario que DENDRO, pero con
    # la estructura que tú has pasado a mano.
    L1_MAP = L2_MAP = None
    EXPERT_GROUPS = None
    OTHER_L1_LABEL = None
else:
    raise ValueError("JERARQUIA debe ser 'H2', 'H3', 'DENDRO' o 'TREE'.")

# =========================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================================================
print("Cargando y filtrando datos...")
df = pd.read_csv(FILE_PATH)

metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']
feature_cols = [c for c in df.columns if c not in metadata_cols]

X_raw = df[feature_cols].fillna(0)

# --- Añadir dimensiones latentes del GPLVM (alineadas por orden de fila) ---
if USAR_LATENTES_GPLVM and os.path.exists(GPLVM_PATH):
    lat = pd.read_csv(GPLVM_PATH)
    lat_cols = [c for c in lat.columns if c.startswith('Latent_')]
    if len(lat) == len(df):
        X_raw = pd.concat([X_raw.reset_index(drop=True),
                           lat[lat_cols].reset_index(drop=True)], axis=1)
        feature_cols = X_raw.columns.tolist()
        print(f"  Añadidas {len(lat_cols)} dimensiones latentes del GPLVM.")
    else:
        print("  AVISO: filas GPLVM no coinciden con el dataset; se omiten latentes.")

y_specie_all = df['Target_Specie']
status = df['Extant/Extinct '].astype(str)

# Asignación de niveles (solo para H2/H3; en DENDRO la estructura es un árbol)
if JERARQUIA in ("H2", "H3"):
    y_L1_all = df['Target_Specie'].map(L1_MAP)
    y_L2_all = df['Target_Specie'].map(L2_MAP)  # NaN para Theropithecus
else:
    y_L1_all = y_L2_all = pd.Series([None] * len(df))

# --- Para DENDRO: columnas latentes del GPLVM sobre las que se hace el clustering.
#     Se cargan SIEMPRE (independientemente de USAR_LATENTES_GPLVM, que controla si
#     se usan como features). El dendrograma se reconstruye DENTRO de cada fold con
#     solo el train, evitando fuga de datos.
LAT_DENDRO = None
if JERARQUIA == "DENDRO":
    if not os.path.exists(GPLVM_PATH):
        raise FileNotFoundError(
            f"JERARQUIA=DENDRO requiere el espacio latente en {GPLVM_PATH}.")
    _lat = pd.read_csv(GPLVM_PATH)
    _lat_cols = [c for c in _lat.columns if c.startswith('Latent_')]
    if len(_lat) != len(df):
        raise ValueError("Las filas del GPLVM no coinciden con el dataset; "
                         "no se puede construir el dendrograma alineado.")
    LAT_DENDRO = _lat[_lat_cols].reset_index(drop=True)
    print(f"  Dendrograma DENDRO sobre {len(_lat_cols)} dimensiones latentes "
          f"({DENDRO_LINKAGE}).")

# Estandarización global
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

mask_extant = status.str.contains('Extant', case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

# Sets de Entrenamiento (Vivos)
X_train = X_scaled[mask_extant].reset_index(drop=True)
y_train_specie_raw = y_specie_all[mask_extant].reset_index(drop=True)
y_train_L1_raw = y_L1_all[mask_extant].reset_index(drop=True)
y_train_L2_raw = y_L2_all[mask_extant].reset_index(drop=True)

# Latentes alineadas al train (para el dendrograma DENDRO)
LAT_TRAIN = LAT_DENDRO[mask_extant.values].reset_index(drop=True) if LAT_DENDRO is not None else None

# Sets de Proyección (Fósiles)
X_project = X_scaled[mask_extinct]
y_project_ids = df.loc[mask_extinct, 'Specimen']

print(f"Muestras de entrenamiento (Extant): {len(X_train)}")
print(f"Muestras a proyectar (Extinct): {len(X_project)}")

# =========================================================
# 2. MOTOR BAYESIANO (FUNCIÓN REUTILIZABLE)
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
            raise ValueError("Prior no soportado.")

        alpha = pt.concatenate([pt.zeros(1), alpha_rest])
        beta = pt.concatenate([pt.zeros((n_features, 1)), beta_rest], axis=1)

        mu = pm.math.dot(X_data, beta) + alpha
        p = pm.Deterministic('p', pm.math.softmax(mu, axis=1))

        y_obs = pm.Categorical('y_obs', p=p, observed=y_data)

        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=42, progressbar=progressbar)
        pm.compute_log_likelihood(idata)

    return model, idata


def predecir_proba_bayesiano(model, idata, X_new):
    """Devuelve la matriz completa de probabilidades medias a posteriori."""
    with model:
        pm.set_data({'X': X_new, 'y': np.zeros(len(X_new), dtype=int)})
        ppc = pm.sample_posterior_predictive(
            idata, predictions=True, var_names=['p'],
            random_seed=42, progressbar=False)
    p_samples = ppc.predictions['p'].values
    p_mean = p_samples.reshape(-1, len(X_new), p_samples.shape[-1]).mean(axis=0)
    return p_mean


# =========================================================
# 2b. MOTOR DE ÁRBOL BINARIO DERIVADO DEL DENDROGRAMA (DENDRO)
#     Construye un árbol binario a partir del clustering jerárquico
#     de los centroides por especie en el espacio latente GPLVM.
#     Cada NODO INTERNO es un clasificador binario bayesiano que
#     decide "rama izquierda vs rama derecha". La clasificación baja
#     por el árbol multiplicando probabilidades hasta llegar a una
#     hoja (una especie). Esto es "ir eligiendo hasta el final".
# =========================================================
def construir_arbol_dendro(X_lat_tr, y_specie_tr, linkage_method, metric):
    """Construye el árbol binario a partir de los centroides por especie.

    X_lat_tr     : DataFrame con las columnas latentes (Latent_*) del train
    y_specie_tr  : Series con la especie de cada fila del train
    Devuelve un dict-árbol: nodo hoja {'leaf':True,'species':...} o nodo interno
    {'leaf':False,'left':...,'right':...,'members':[...]}.
    """
    dfc = X_lat_tr.copy()
    dfc['__sp__'] = y_specie_tr.values
    cent = dfc.groupby('__sp__').mean()
    species = cent.index.tolist()
    if len(species) == 1:
        return {'leaf': True, 'species': species[0], 'members': species}
    Z = linkage(cent.values, method=linkage_method, metric=metric)
    root = to_tree(Z)

    def build(node):
        if node.is_leaf():
            sp = species[node.id]
            return {'leaf': True, 'species': sp, 'members': [sp]}
        L, R = build(node.left), build(node.right)
        return {'leaf': False, 'left': L, 'right': R,
                'members': L['members'] + R['members']}
    return build(root)


def construir_arbol_fijo(spec, especies_presentes=None):
    """Construye el árbol binario a partir de una LISTA ANIDADA escrita a mano.

    spec : o bien un string (nombre de especie = hoja), o bien una lista/tupla de
           exactamente 2 elementos [izquierda, derecha], cada uno recursivamente
           un spec. Reproduce la forma del dendrograma con paréntesis anidados.
    especies_presentes : conjunto opcional de especies realmente disponibles en el
           train del fold. Si se pasa, las hojas ausentes se podan y los nodos que
           queden con un solo hijo se colapsan (evita romper la cascada en LOOCV).

    Devuelve la MISMA estructura de dict que construir_arbol_dendro:
      hoja  -> {'leaf':True,  'species':sp, 'members':[sp]}
      nodo  -> {'leaf':False, 'left':L, 'right':R, 'members':[...]}.
    """
    def build(s):
        # Hoja: nombre de especie
        if isinstance(s, str):
            if especies_presentes is not None and s not in especies_presentes:
                return None  # especie ausente en este fold -> se poda
            return {'leaf': True, 'species': s, 'members': [s]}
        # Nodo interno: lista/tupla de 2 specs
        if not isinstance(s, (list, tuple)) or len(s) != 2:
            raise ValueError(
                "Cada nodo interno de ARBOL_FIJO debe ser una lista/tupla de 2 "
                f"elementos [izquierda, derecha]; recibido: {s!r}")
        L = build(s[0])
        R = build(s[1])
        # Poda: si falta una rama, el nodo se colapsa en la rama que quede.
        if L is None and R is None:
            return None
        if L is None:
            return R
        if R is None:
            return L
        return {'leaf': False, 'left': L, 'right': R,
                'members': L['members'] + R['members']}

    tree = build(spec)
    if tree is None:
        raise ValueError("ARBOL_FIJO se quedó vacío tras la poda; revisa los nombres "
                         "de especie y que coincidan con 'Target_Specie'.")
    return tree


def _iter_nodos_internos(tree, path='root'):
    """Recorre los nodos internos del árbol (cada uno = un clasificador)."""
    if tree['leaf']:
        return
    yield path, tree
    yield from _iter_nodos_internos(tree['left'], path + '.L')
    yield from _iter_nodos_internos(tree['right'], path + '.R')


def entrenar_arbol_dendro(X_tr, y_specie_tr, tree, tipo_prior,
                          draws=DRAWS, tune=TUNE, progressbar=True):
    """Entrena un clasificador binario bayesiano en CADA nodo interno del árbol.

    En cada nodo, la etiqueta es 0 si la especie de la muestra cae en la rama
    izquierda (members del hijo izquierdo) y 1 si cae en la derecha. Solo se
    entrena con las muestras cuyas especies pertenecen al nodo.
    """
    X_tr = X_tr.reset_index(drop=True)
    y_specie_tr = y_specie_tr.reset_index(drop=True)
    nodos = {}
    for path, node in _iter_nodos_internos(tree):
        izq = set(node['left']['members'])
        der = set(node['right']['members'])
        mask = y_specie_tr.isin(izq | der).values
        y_bin = y_specie_tr[mask].isin(der).astype(int).values  # 0=izq, 1=der
        # Si en este fold falta una de las dos ramas, el nodo es degenerado:
        # se marca con la rama presente para no romper la cascada.
        if len(np.unique(y_bin)) < 2:
            nodos[path] = {'degenerate': True,
                           'forced': 1 if y_bin.mean() > 0.5 else 0}
            continue
        model, idata = construir_y_muestrear_modelo(
            X_tr[mask].values, y_bin, 2, tipo_prior,
            draws=draws, tune=tune, progressbar=progressbar)
        nodos[path] = {'degenerate': False, 'model': model, 'idata': idata}
    return {'tree': tree, 'nodos': nodos}


def clasificar_cascada_dendro(h, X_new):
    """Baja por el árbol nodo a nodo multiplicando P(rama) hasta cada hoja.

    Devuelve (lista_especies_predichas, lista_prob_conjunta). La probabilidad
    de una especie es el producto de las probabilidades de las ramas tomadas
    desde la raíz hasta su hoja.
    """
    tree, nodos = h['tree'], h['nodos']
    n = len(X_new)

    # Pre-calcula P(derecha) en cada nodo interno para todas las muestras
    p_der = {}
    for path, node in _iter_nodos_internos(tree):
        info = nodos[path]
        if info['degenerate']:
            p_der[path] = np.full(n, float(info['forced']))
        else:
            p = predecir_proba_bayesiano(info['model'], info['idata'], X_new)
            p_der[path] = p[:, 1]  # columna 1 = rama derecha

    # Probabilidad conjunta de cada hoja (especie) por recursión
    species_probs = [dict() for _ in range(n)]

    def descender(node, path, prob_acum):
        if node['leaf']:
            for i in range(n):
                species_probs[i][node['species']] = prob_acum[i]
            return
        pd_ = p_der[path]
        descender(node['left'],  path + '.L', prob_acum * (1.0 - pd_))
        descender(node['right'], path + '.R', prob_acum * pd_)

    descender(tree, 'root', np.ones(n))

    preds, confs = [], []
    for i in range(n):
        best = max(species_probs[i], key=species_probs[i].get)
        preds.append(best)
        confs.append(species_probs[i][best])
    return preds, confs, species_probs


# =========================================================
# 3. ENTRENAMIENTO DE LA JERARQUÍA COMPLETA (REUTILIZABLE)
#    Devuelve todos los modelos/encoders necesarios para
#    clasificar en cascada. Se llama tanto para el modelo
#    final como dentro de cada fold del LOOCV.
# =========================================================
def entrenar_jerarquia(X_tr, y_specie_tr, y_L1_tr, y_L2_tr,
                       tipo_prior, draws=DRAWS, tune=TUNE, progressbar=True):
    """Entrena los 3 niveles sobre el subconjunto de entrenamiento dado.

    X_tr            : DataFrame de features (estandarizadas)
    y_specie_tr     : Series con la especie real
    y_L1_tr, y_L2_tr: Series con las etiquetas de nivel 1 y 2
    """
    X_tr = X_tr.reset_index(drop=True)
    y_specie_tr = y_specie_tr.reset_index(drop=True)
    y_L1_tr = y_L1_tr.reset_index(drop=True)
    y_L2_tr = y_L2_tr.reset_index(drop=True)

    # ---- Nivel 1: Theropithecus vs Other_L1 ----
    le_L1 = LabelEncoder()
    y_L1_enc = le_L1.fit_transform(y_L1_tr)
    model_L1, idata_L1 = construir_y_muestrear_modelo(
        X_tr.values, y_L1_enc, len(le_L1.classes_), tipo_prior,
        draws=draws, tune=tune, progressbar=progressbar)

    # ---- Nivel 2: Grupo_Papio vs Grupo_Other3 (solo sobre Other_L1) ----
    mask_L2 = (y_L1_tr == "Other_L1")
    le_L2 = LabelEncoder()
    y_L2_enc = le_L2.fit_transform(y_L2_tr[mask_L2])
    model_L2, idata_L2 = construir_y_muestrear_modelo(
        X_tr[mask_L2].values, y_L2_enc, len(le_L2.classes_), tipo_prior,
        draws=draws, tune=tune, progressbar=progressbar)

    # ---- Nivel 3: expertos de especie por grupo ----
    species_models = {}
    species_encoders = {}
    for group in EXPERT_GROUPS:
        mask_L3 = (y_L2_tr == group)
        le_sp = LabelEncoder()
        y_sp_enc = le_sp.fit_transform(y_specie_tr[mask_L3])
        species_encoders[group] = le_sp
        model_L3, idata_L3 = construir_y_muestrear_modelo(
            X_tr[mask_L3].values, y_sp_enc, len(le_sp.classes_), tipo_prior,
            draws=draws, tune=tune, progressbar=progressbar)
        species_models[group] = (model_L3, idata_L3)

    return {
        'le_L1': le_L1, 'model_L1': model_L1, 'idata_L1': idata_L1,
        'le_L2': le_L2, 'model_L2': model_L2, 'idata_L2': idata_L2,
        'species_models': species_models, 'species_encoders': species_encoders,
    }


def clasificar_cascada(h, X_new):
    """Clasifica en cascada usando la regla de probabilidad total de 3 niveles.

    h     : dict devuelto por entrenar_jerarquia
    X_new : array (n, d) de muestras a clasificar
    Devuelve (lista_especies_predichas, lista_prob_maxima).
    """
    le_L1, le_L2 = h['le_L1'], h['le_L2']
    species_encoders = h['species_encoders']

    all_species = ["Theropithecus gelada"]
    all_species.extend(species_encoders["Grupo_Papio"].classes_)
    all_species.extend(species_encoders["Grupo_Other3"].classes_)

    idx_L1_thero = le_L1.transform(["Theropithecus"])[0]
    idx_L1_other = le_L1.transform(["Other_L1"])[0]
    idx_L2_papio = le_L2.transform(["Grupo_Papio"])[0]
    idx_L2_other = le_L2.transform(["Grupo_Other3"])[0]

    p_L1 = predecir_proba_bayesiano(h['model_L1'], h['idata_L1'], X_new)
    p_L2 = predecir_proba_bayesiano(h['model_L2'], h['idata_L2'], X_new)
    p_L3_papio = predecir_proba_bayesiano(
        species_models_ref(h, "Grupo_Papio"), h['species_models']["Grupo_Papio"][1], X_new)
    p_L3_other = predecir_proba_bayesiano(
        species_models_ref(h, "Grupo_Other3"), h['species_models']["Grupo_Other3"][1], X_new)

    preds, confs = [], []
    for i in range(len(X_new)):
        probs = {sp: 0.0 for sp in all_species}
        probs["Theropithecus gelada"] += p_L1[i, idx_L1_thero]
        for k, sp in enumerate(species_encoders["Grupo_Papio"].classes_):
            probs[sp] += p_L1[i, idx_L1_other] * p_L2[i, idx_L2_papio] * p_L3_papio[i, k]
        for k, sp in enumerate(species_encoders["Grupo_Other3"].classes_):
            probs[sp] += p_L1[i, idx_L1_other] * p_L2[i, idx_L2_other] * p_L3_other[i, k]
        best_sp = max(probs, key=probs.get)
        preds.append(best_sp)
        confs.append(probs[best_sp])
    return preds, confs


def species_models_ref(h, group):
    """Pequeño helper para legibilidad: devuelve el modelo del grupo."""
    return h['species_models'][group][0]


# =========================================================
# 4. LOOCV EXACTO (REENTRENA LA JERARQUÍA EN CADA FOLD)
# =========================================================
# --- Validación del árbol fijo (solo TREE): comprobar que los nombres de las
#     hojas coinciden con las especies extant del dataset, ni de más ni de menos.
if JERARQUIA == "TREE":
    def _hojas_de_spec(s):
        if isinstance(s, str):
            return {s}
        return _hojas_de_spec(s[0]) | _hojas_de_spec(s[1])
    hojas_arbol = _hojas_de_spec(ARBOL_FIJO)
    especies_extant = set(y_train_specie_raw.unique())
    faltan_en_datos = hojas_arbol - especies_extant   # están en el árbol pero no en los datos
    faltan_en_arbol = especies_extant - hojas_arbol   # están en los datos pero no en el árbol
    if faltan_en_datos:
        print(f"  AVISO TREE: hojas del árbol que NO aparecen en los extant "
              f"(se podarán por fold): {sorted(faltan_en_datos)}")
    if faltan_en_arbol:
        raise ValueError(
            "ARBOL_FIJO no incluye todas las especies extant del dataset. "
            f"Faltan en el árbol: {sorted(faltan_en_arbol)}. "
            "Cada especie viva debe ser una hoja del árbol.")
    print(f"  Árbol FIJO validado: {len(hojas_arbol)} hojas, "
          f"{len(hojas_arbol) - 1} nodos internos (submodelos).")

print("\n" + "=" * 50)
print(" LOOCV EXACTO (3 NIVELES) — esto reentrena la jerarquía N veces")
print(f"  Checkpoint en tiempo real: '{CHECKPOINT_LOO}'")
print("=" * 50)

loo = LeaveOneOut()
folds = list(loo.split(X_train))
n = len(X_train)

# ---- Reanudación: cargar folds ya calculados ----
resultados = {}
if os.path.exists(CHECKPOINT_LOO):
    with open(CHECKPOINT_LOO) as f:
        saved = json.load(f)
    resultados = {int(k): v for k, v in saved.items()}
    print(f"  Checkpoint encontrado: {len(resultados)}/{n} folds ya resueltos. "
          f"Continuando con los restantes...")

folds_pendientes = [(fold, te) for fold, (tr, te) in enumerate(folds, start=1)
                    if fold not in resultados]

# Loop SECUENCIAL: cada fold reentrena la jerarquía completa (4 modelos) y PyMC
# paraleliza las CHAINS cadenas de cada submodelo internamente. tqdm da la barra
# de progreso; el JSON se actualiza tras CADA fold.
for fold, te_idx in tqdm(folds_pendientes, total=len(folds_pendientes),
                         desc="Progreso LOOCV", unit="fold"):
    i = te_idx[0]
    tr_idx = np.setdiff1d(np.arange(n), te_idx)

    X_tr = X_train.iloc[tr_idx]
    y_sp_tr = y_train_specie_raw.iloc[tr_idx]
    y_L1_tr = y_train_L1_raw.iloc[tr_idx]
    y_L2_tr = y_train_L2_raw.iloc[tr_idx]
    X_te = X_train.iloc[te_idx].values

    if JERARQUIA == "DENDRO":
        # Reconstruir el dendrograma SOLO con el train del fold (sin fuga).
        tree_fold = construir_arbol_dendro(
            LAT_TRAIN.iloc[tr_idx], y_sp_tr, DENDRO_LINKAGE, DENDRO_METRIC)
        h_fold = entrenar_arbol_dendro(
            X_tr, y_sp_tr, tree_fold, TIPO_PRIOR,
            draws=DRAWS_CV, tune=TUNE_CV, progressbar=False)
        pred_sp, _, _ = clasificar_cascada_dendro(h_fold, X_te)
    elif JERARQUIA == "TREE":
        # Árbol FIJO. No se recalcula nada; solo se poda a las especies presentes
        # en el train de este fold (en LOOCV puede faltar 1 muestra de una especie,
        # pero la especie sigue presente si tiene >1 individuo; la poda cubre el
        # caso límite de especies con una sola muestra).
        tree_fold = construir_arbol_fijo(
            ARBOL_FIJO, especies_presentes=set(y_sp_tr.unique()))
        h_fold = entrenar_arbol_dendro(
            X_tr, y_sp_tr, tree_fold, TIPO_PRIOR,
            draws=DRAWS_CV, tune=TUNE_CV, progressbar=False)
        pred_sp, _, _ = clasificar_cascada_dendro(h_fold, X_te)
    else:
        h_fold = entrenar_jerarquia(
            X_tr, y_sp_tr, y_L1_tr, y_L2_tr, TIPO_PRIOR,
            draws=DRAWS_CV, tune=TUNE_CV, progressbar=False)
        pred_sp, _ = clasificar_cascada(h_fold, X_te)

    resultados[fold] = {
        'idx': int(i),
        'y_true': str(y_train_specie_raw.iloc[i]),
        'y_pred': str(pred_sp[0]),
    }

    # Guardado en tiempo real: actualiza el JSON tras CADA fold completado
    with open(CHECKPOINT_LOO, "w") as f:
        json.dump({str(k): v for k, v in resultados.items()}, f, indent=2)

if folds_pendientes:
    print(f"  -> Todos los folds completados y guardados en: '{CHECKPOINT_LOO}'")

# ---- Reconstruir arrays en el orden de los folds ----
y_true_loo = np.array([resultados[fold]['y_true'] for fold in sorted(resultados)])
y_pred_loo = np.array([resultados[fold]['y_pred'] for fold in sorted(resultados)])

# ---- Métricas LOOCV (las mismas para todos los modelos del pipeline) ----
acc_loo = accuracy_score(y_true_loo, y_pred_loo)
bacc_loo = balanced_accuracy_score(y_true_loo, y_pred_loo)
f1_loo = f1_score(y_true_loo, y_pred_loo, average='macro')

print("\n--- MÉTRICAS LOOCV (nivel especie) ---")
print(f"Accuracy           : {acc_loo:.2%}")
print(f"Balanced accuracy  : {bacc_loo:.2%}")
print(f"F1 macro           : {f1_loo:.2%}")

report_loo = classification_report(y_true_loo, y_pred_loo)
print("\n" + report_loo)

# Guardar resumen de métricas como imagen
texto_loo = (f"=== MÉTRICAS LOOCV — JERÁRQUICO 3 NIVELES ({TIPO_PRIOR.upper()}) ===\n\n"
             f"Accuracy          : {acc_loo:.4f}\n"
             f"Balanced accuracy : {bacc_loo:.4f}\n"
             f"F1 macro          : {f1_loo:.4f}\n\n"
             + report_loo)
fig_txt, ax_txt = plt.subplots(figsize=(10, 8))
ax_txt.axis('off')
ax_txt.text(0.01, 0.99, texto_loo, fontsize=11, fontfamily='monospace',
            va='top', ha='left', transform=ax_txt.transAxes)
plt.savefig(f'metricas_loocv_jerarquia3niveles_{TIPO_PRIOR}.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"-> Guardado: 'metricas_loocv_jerarquia3niveles_{TIPO_PRIOR}.png'")

# ---- Matriz de confusión LOO ----
labels_sorted = np.unique(np.concatenate([y_true_loo, y_pred_loo]))
cm_loo = confusion_matrix(y_true_loo, y_pred_loo, labels=labels_sorted)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_loo, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.title(f'Matriz de Confusión LOOCV — Jerárquico 3 Niveles ({TIPO_PRIOR})')
plt.xlabel('Predicción (LOO)')
plt.ylabel('Realidad')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"matriz_confusion_loocv_3niveles_{TIPO_PRIOR}.png", dpi=300)
plt.close()
print(f"-> Guardado: 'matriz_confusion_loocv_3niveles_{TIPO_PRIOR}.png'")

# Guardar predicciones LOO en CSV (para construir la tabla comparativa luego)
pd.DataFrame({
    'y_true': y_true_loo,
    'y_pred_loo': y_pred_loo,
}).to_csv(f"predicciones_loocv_3niveles_{TIPO_PRIOR}.csv", index=False)
print(f"-> Guardado: 'predicciones_loocv_3niveles_{TIPO_PRIOR}.csv'")

# =========================================================
# 5. MODELO FINAL: entrenar con TODOS los extant
#    (este es el modelo que se usa para proyectar fósiles)
# =========================================================
print("\n" + "=" * 50)
print(" ENTRENANDO MODELO FINAL CON TODOS LOS EXTANT")
print("=" * 50)

if JERARQUIA == "DENDRO":
    # Árbol final con todos los extant + figura del dendrograma para la memoria.
    tree_final = construir_arbol_dendro(
        LAT_TRAIN, y_train_specie_raw, DENDRO_LINKAGE, DENDRO_METRIC)
    h_final = entrenar_arbol_dendro(
        X_train, y_train_specie_raw, tree_final, TIPO_PRIOR,
        draws=DRAWS, tune=TUNE, progressbar=True)

    # Figura del dendrograma (centroides por especie en el espacio latente)
    _dfc = LAT_TRAIN.copy()
    _dfc['__sp__'] = y_train_specie_raw.values
    _cent = _dfc.groupby('__sp__').mean()
    _Z = linkage(_cent.values, method=DENDRO_LINKAGE, metric=DENDRO_METRIC)
    plt.figure(figsize=(11, 6))
    dendrogram(_Z, labels=_cent.index.tolist(), leaf_rotation=45)
    plt.title(f"Dendrograma GPLVM ({DENDRO_LINKAGE}) — jerarquía usada en cascada")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(f"dendrograma_gplvm_{DENDRO_LINKAGE}_{TIPO_PRIOR}.png", dpi=300)
    plt.close()
    print(f"-> Guardado: 'dendrograma_gplvm_{DENDRO_LINKAGE}_{TIPO_PRIOR}.png'")
elif JERARQUIA == "TREE":
    # Árbol FIJO final con todos los extant. Un submodelo por nodo interno.
    tree_final = construir_arbol_fijo(
        ARBOL_FIJO, especies_presentes=set(y_train_specie_raw.unique()))
    h_final = entrenar_arbol_dendro(
        X_train, y_train_specie_raw, tree_final, TIPO_PRIOR,
        draws=DRAWS, tune=TUNE, progressbar=True)
    n_nodos = sum(1 for _ in _iter_nodos_internos(tree_final))
    print(f"  Entrenados {n_nodos} submodelos binarios (uno por nodo interno).")

    # Figura del árbol fijo (dibujo de la estructura que has pasado a mano).
    def _dibujar_arbol_fijo(tree, fname):
        # Asigna posiciones x a las hojas (en orden) y dibuja el árbol como
        # un dendrograma sin escala de distancia (solo topología).
        hojas = tree['members']
        x_de_hoja = {sp: k for k, sp in enumerate(hojas)}
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(hojas)), 6))
        niveles = {}  # asigna una altura a cada nodo por profundidad inversa

        def alto(node):
            if node['leaf']:
                return 0
            return 1 + max(alto(node['left']), alto(node['right']))

        def dibujar(node):
            if node['leaf']:
                x = x_de_hoja[node['species']]
                return x, 0.0
            xl, yl = dibujar(node['left'])
            xr, yr = dibujar(node['right'])
            y = max(yl, yr) + 1.0
            xm = (xl + xr) / 2.0
            ax.plot([xl, xl], [yl, y], color='tab:blue')
            ax.plot([xr, xr], [yr, y], color='tab:blue')
            ax.plot([xl, xr], [y, y], color='tab:blue')
            return xm, y

        dibujar(tree)
        ax.set_xticks(range(len(hojas)))
        ax.set_xticklabels(hojas, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title("Árbol FIJO (definido a mano) — jerarquía usada en cascada")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    _dibujar_arbol_fijo(tree_final, f"arbol_fijo_{TIPO_PRIOR}.png")
    print(f"-> Guardado: 'arbol_fijo_{TIPO_PRIOR}.png'")
else:
    h_final = entrenar_jerarquia(
        X_train, y_train_specie_raw, y_train_L1_raw, y_train_L2_raw, TIPO_PRIOR,
        draws=DRAWS, tune=TUNE, progressbar=True)

# =========================================================
# 6. PROYECCIÓN FÓSILES (PROBABILIDAD CONJUNTA 3 NIVELES)
#    Usa el modelo final entrenado con todos los extant.
# =========================================================
if len(X_project) > 0:
    print("\n" + "=" * 50)
    print(" PROYECTANDO FÓSILES (PROBABILIDAD CONJUNTA)")
    print("=" * 50)

    fossil_predictions = []
    prob_rows = []

    if JERARQUIA in ("DENDRO", "TREE"):
        # Cascada por el árbol binario: P(especie) = producto de P(rama) tomadas.
        all_extant_species_list = sorted(h_final['tree']['members'])
        preds_sp, _, sp_probs_list = clasificar_cascada_dendro(h_final, X_project.values)
        for i in range(len(X_project)):
            specimen_id = y_project_ids.iloc[i]
            row = {'Specimen_Extinct': specimen_id}
            sp_probs = sp_probs_list[i]
            for sp in all_extant_species_list:
                row[f"P_{sp.replace(' ', '_')}"] = np.round(sp_probs.get(sp, 0.0), 4)
            best_sp = preds_sp[i]
            row['Best_Extant_Specie'] = best_sp
            row['Best_Extant_Prob'] = np.round(sp_probs[best_sp], 4)
            prob_rows.append(row)
            fossil_predictions.append({'Specimen_Extinct': specimen_id, 'Pred_Specie': best_sp})
    else:
        le_L1, le_L2 = h_final['le_L1'], h_final['le_L2']
        species_encoders = h_final['species_encoders']

        p_mat_L1 = predecir_proba_bayesiano(h_final['model_L1'], h_final['idata_L1'], X_project.values)
        p_mat_L2 = predecir_proba_bayesiano(h_final['model_L2'], h_final['idata_L2'], X_project.values)
        p_mat_L3_papio = predecir_proba_bayesiano(
            h_final['species_models']["Grupo_Papio"][0],
            h_final['species_models']["Grupo_Papio"][1], X_project.values)
        p_mat_L3_other = predecir_proba_bayesiano(
            h_final['species_models']["Grupo_Other3"][0],
            h_final['species_models']["Grupo_Other3"][1], X_project.values)

        all_extant_species_list = ["Theropithecus gelada"]
        all_extant_species_list.extend(species_encoders["Grupo_Papio"].classes_)
        all_extant_species_list.extend(species_encoders["Grupo_Other3"].classes_)

        idx_L1_thero = le_L1.transform(["Theropithecus"])[0]
        idx_L1_other = le_L1.transform(["Other_L1"])[0]
        idx_L2_papio = le_L2.transform(["Grupo_Papio"])[0]
        idx_L2_other = le_L2.transform(["Grupo_Other3"])[0]

        for i in range(len(X_project)):
            specimen_id = y_project_ids.iloc[i]
            row = {'Specimen_Extinct': specimen_id}

            species_probs = {sp: 0.0 for sp in all_extant_species_list}
            species_probs["Theropithecus gelada"] += p_mat_L1[i, idx_L1_thero]
            for k, sp in enumerate(species_encoders["Grupo_Papio"].classes_):
                species_probs[sp] += p_mat_L1[i, idx_L1_other] * p_mat_L2[i, idx_L2_papio] * p_mat_L3_papio[i, k]
            for k, sp in enumerate(species_encoders["Grupo_Other3"].classes_):
                species_probs[sp] += p_mat_L1[i, idx_L1_other] * p_mat_L2[i, idx_L2_other] * p_mat_L3_other[i, k]

            for sp in all_extant_species_list:
                row[f"P_{sp.replace(' ', '_')}"] = np.round(species_probs[sp], 4)

            best_sp = max(species_probs, key=species_probs.get)
            row['Best_Extant_Specie'] = best_sp
            row['Best_Extant_Prob'] = np.round(species_probs[best_sp], 4)
            prob_rows.append(row)
            fossil_predictions.append({'Specimen_Extinct': specimen_id, 'Pred_Specie': best_sp})

    probs_table = pd.DataFrame(prob_rows)
    output_probs_csv = f"predicciones_fosiles_{JERARQUIA}_probs_{TIPO_PRIOR}.csv"
    probs_table.to_csv(output_probs_csv, index=False)
    print(f"📁 Guardado: {output_probs_csv}")

    # ---- Matriz de elecciones (Extinct × Extant) ----
    extinct_species_labels = df.loc[mask_extinct, 'Target_Specie'].reset_index(drop=True)
    extinct_species_unique = extinct_species_labels.unique()
    count_matrix = pd.DataFrame(0, index=sorted(extinct_species_unique),
                                columns=all_extant_species_list)
    for i, pred_info in enumerate(fossil_predictions):
        extinct_sp = extinct_species_labels.iloc[i]
        chosen_extant = pred_info['Pred_Specie']
        if chosen_extant in count_matrix.columns:
            count_matrix.loc[extinct_sp, chosen_extant] += 1

    count_matrix.to_csv(f"matriz_elecciones_extinct_extant_{JERARQUIA}_{TIPO_PRIOR}.csv")
    plt.figure(figsize=(12, 8))
    sns.heatmap(count_matrix, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, linecolor='lightgrey')
    plt.title(f"Matriz de Elecciones: Extinct × Extant ({JERARQUIA}, {TIPO_PRIOR.upper()})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"matriz_elecciones_extinct_extant_{JERARQUIA}_{TIPO_PRIOR}.png", dpi=300)
    plt.close()
    print(f"-> Guardado: 'matriz_elecciones_extinct_extant_{JERARQUIA}_{TIPO_PRIOR}.png'")

print("\n✅ Ejecución completada: LOOCV exacto + modelo final + proyección de fósiles.")