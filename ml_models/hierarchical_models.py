import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

# Nuevas librerías para la visualización del árbol
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout # Opcional, pero usaremos un método nativo

from sklearn.model_selection import (
    LeaveOneOut, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, accuracy_score,
    f1_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# ⚙️ CONFIGURACIÓN GENERAL
# =========================================================
JERARQUIA = "TREE"  # Opciones: "H3" (original por niveles) o "TREE" (árbol binario arbitrario)

# --- Árbol FIJO (solo se usa si JERARQUIA == "TREE") ---
# --- Árbol FIJO (solo se usa si JERARQUIA == "TREE") ---
ARBOL_FIJO = [
    "Theropithecus gelada",
    [
        [
            "Papio ursinus",
            [
                "Lophocebus aterrimus",
                [
                    "Papio anubis",
                    "Papio hamadryas",
                ],
            ],
        ],
        [
            "Cercocebus atys",
            "Macaca sylvanus",
        ],
    ],
]

# Configuración H3 Original (Se mantiene por compatibilidad)
L1_SETS = {
    "Theropithecus": ["Theropithecus gelada"],
    "Other_L1": ["Papio anubis", "Papio hamadryas", "Papio ursinus", "Lophocebus aterrimus", "Cercocebus atys", "Macaca sylvanus"]
}
L2_SETS = {
    "Papio": ["Papio anubis", "Papio hamadryas", "Papio ursinus", "Lophocebus aterrimus"],
    "Other_3": ["Cercocebus atys", "Macaca sylvanus"]
}
L3_SETS = L2_SETS.copy()
L3_SETS["Theropithecus"] = ["Theropithecus gelada"]

def get_l1_label(sp): return next((k for k, v in L1_SETS.items() if sp in v), np.nan)
def get_l2_label(sp): return next((k for k, v in L2_SETS.items() if sp in v), np.nan)
def get_l3_label(sp): return next((k for k, v in L3_SETS.items() if sp in v), np.nan)

# =========================================================
# 1. CARGA DE DATOS
# =========================================================
LATENT_PATH   = "src/espacio_latente_gplvm_1.csv" 
ORIGINAL_PATH = "src/dataset_con_grupos.csv"

metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']

# Cargar Datos Latentes
try:
    df_latent = pd.read_csv(LATENT_PATH)
    print(f"✅ Datos latentes cargados: {df_latent.shape}")
except FileNotFoundError as e:
    print(f"❌ Error al leer espacio latente: {e}")
    exit()

latent_feature_cols = [c for c in df_latent.columns if c.startswith('Latent_')]
X_latent_all = df_latent[latent_feature_cols]

# Cargar Datos Originales
try:
    df_orig = pd.read_csv(ORIGINAL_PATH)
    print(f"✅ Datos originales cargados: {df_orig.shape}")
except FileNotFoundError as e:
    print(f"❌ Error al leer datos originales: {e}")
    exit()

orig_feature_cols = [c for c in df_orig.columns if c not in metadata_cols]
scaler = StandardScaler()
X_orig_scaled = pd.DataFrame(
    scaler.fit_transform(df_orig[orig_feature_cols].fillna(0)),
    columns=orig_feature_cols
)

# Combinar Latentes y Originales
X_full = pd.concat([X_latent_all.reset_index(drop=True), X_orig_scaled.reset_index(drop=True)], axis=1)

status       = df_orig['Extant/Extinct '].astype(str)
mask_extant  = status.str.contains('Extant',  case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

y_specie_all = df_orig['Target_Specie']

# Conjuntos de Entrenamiento
X_train = X_full[mask_extant].reset_index(drop=True)
y_train_sp = y_specie_all[mask_extant].reset_index(drop=True)

# Conjuntos de Proyección (Fósiles)
X_project = X_full[mask_extinct].reset_index(drop=True)
y_project_ids = df_orig.loc[mask_extinct, 'Specimen'].reset_index(drop=True)

print(f"🧬 Muestras de entrenamiento: {len(X_train)}")
print(f"🦴 Fósiles a proyectar: {len(y_project_ids)}")

# =========================================================
# 2. MODELOS BASE (SCIKIT-LEARN)
# =========================================================
base_models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
}

param_grids = {
    "Random Forest": {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10]},
}

# =========================================================
# 3. MOTOR DE ÁRBOL BINARIO (Para JERARQUIA == "TREE")
# =========================================================
def construir_arbol_fijo(spec, especies_presentes=None):
    def build(s):
        if isinstance(s, str):
            if especies_presentes is not None and s not in especies_presentes:
                return None
            return {'leaf': True, 'species': s, 'members': [s]}
        L, R = build(s[0]), build(s[1])
        if L is None and R is None: return None
        if L is None: return R
        if R is None: return L
        # Nombre único para nodo interno basado en miembros ordenados
        m_sorted = sorted(L['members'] + R['members'])
        node_id = f"Node_{hash(tuple(m_sorted))}"
        return {'leaf': False, 'id': node_id, 'left': L, 'right': R, 'members': m_sorted}
    
    tree = build(spec)
    if tree is None: raise ValueError("Árbol vacío tras la poda.")
    return tree

def _iter_nodos_internos(tree, path='root'):
    if tree['leaf']: return
    yield path, tree
    yield from _iter_nodos_internos(tree['left'], path + '.L')
    yield from _iter_nodos_internos(tree['right'], path + '.R')

def entrenar_arbol_ml(X_tr, y_sp_tr, tree):
    nodos = {}
    for path, node in _iter_nodos_internos(tree):
        izq = set(node['left']['members'])
        der = set(node['right']['members'])
        mask = y_sp_tr.isin(izq | der).values
        
        y_bin = y_sp_tr[mask].isin(der).astype(int).values # 0=Izq, 1=Der
        X_node = X_tr[mask].values
        
        if len(np.unique(y_bin)) < 2:
            nodos[path] = {'degenerate': True, 'forced': 1 if y_bin.mean() > 0.5 else 0}
            continue
            
        best_acc = -1
        best_model = None
        
        # Ajustar splits de CV
        min_class = np.bincount(y_bin).min()
        cv_splits = min(3, min_class)
        cv = StratifiedKFold(n_splits=cv_splits) if cv_splits >= 2 else LeaveOneOut()

        for name, model in base_models.items():
            grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            grid.fit(X_node, y_bin)
            if grid.best_score_ > best_acc:
                best_acc = grid.best_score_
                best_model = grid.best_estimator_
                
        best_model.fit(X_node, y_bin)
        nodos[path] = {'degenerate': False, 'model': best_model}
        
    return {'tree': tree, 'nodos': nodos}

def clasificar_cascada_ml(h, X_new):
    tree, nodos = h['tree'], h['nodos']
    n = len(X_new)
    p_der = {}

    for path, node in _iter_nodos_internos(tree):
        info = nodos[path]
        if info['degenerate']:
            p_der[path] = np.full(n, float(info['forced']))
        else:
            p_der[path] = info['model'].predict_proba(X_new)[:, 1]

    species_probs = [dict() for _ in range(n)]

    def descender(node, path, prob_acum):
        if node['leaf']:
            for i in range(n):
                species_probs[i][node['species']] = prob_acum[i]
            return
        pd_ = p_der[path]
        descender(node['left'], path + '.L', prob_acum * (1.0 - pd_))
        descender(node['right'], path + '.R', prob_acum * pd_)

    descender(tree, 'root', np.ones(n))

    preds, confs = [], []
    for i in range(n):
        best = max(species_probs[i], key=species_probs[i].get)
        preds.append(best)
        confs.append(species_probs[i][best])
    return preds, confs, species_probs

# =========================================================
# 4. FUNCIONES DE VISUALIZACIÓN DEL ÁRBOL (Nuevas)
# =========================================================
def convertir_arbol_a_nx(tree_dict):
    """Convierte la estructura de dict anidados a un DiGraph de NetworkX."""
    G = nx.DiGraph()
    
    def add_nodes(node):
        if node['leaf']:
            # Nodo Hoja: Usamos el nombre de la especie como ID y etiqueta
            label = node['species'].replace(" ", "\n") # Ajuste para etiquetas largas
            G.add_node(node['species'], label=label, type='leaf')
            return node['species']
        
        # Nodo Interno: Usamos su ID único generado en construir_arbol_fijo
        node_id = node['id']
        label = f"Decisión\n({len(node['members'])} sp)"
        G.add_node(node_id, label=label, type='internal')
        
        # Añadir hijos recursivamente y conectar
        left_id = add_nodes(node['left'])
        right_id = add_nodes(node['right'])
        
        G.add_edge(node_id, left_id, side='L')
        G.add_edge(node_id, right_id, side='R')
        return node_id

    add_nodes(tree_dict)
    return G

def calcular_posiciones_jerarquicas(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """Calcula posiciones (x, y) para dibujar un grafo como un árbol jerárquico."""
    if root is None:
        # Asumir que el nodo sin aristas entrantes es la raíz
        roots = [n for n, d in G.in_degree() if d == 0]
        if not roots: raise ValueError("No se encontró una raíz clara en el árbol.")
        root = roots[0]

    def _hierarchy_pos(G, root, left=0.0, right=1.0, y=0.0, pos=None):
        if pos is None: pos = {}
        
        # Posición del nodo actual
        x = (left + right) / 2
        pos[root] = (x, y)
        
        children = list(G.successors(root))
        if children:
            # Separar el espacio horizontal disponible para los hijos
            dx = (right - left) / len(children)
            next_left = left
            for child in children:
                pos = _hierarchy_pos(G, child, next_left, next_left + dx, y - vert_gap, pos=pos)
                next_left += dx
        return pos

    return _hierarchy_pos(G, root, left=0.0, right=width, y=vert_loc)

def guardar_imagen_arbol(tree_dict, filename="estructura_arbol_clasificatorio.png"):
    """Genera y guarda una imagen detallada de la estructura del árbol."""
    print(f"📸 Generando imagen de la estructura del árbol en '{filename}'...")
    
    # 1. Crear grafo
    G = convertir_arbol_a_nx(tree_dict)
    
    # 2. Calcular posiciones
    try:
        # Intentar usar layout nativo optimizado si la estructura es simple
        pos = calcular_posiciones_jerarquicas(G)
    except Exception:
        # Fallback a spring layout si falla la jerarquía
        pos = nx.spring_layout(G)

    # 3. Preparar visualización
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'side')
    
    # Colores según tipo
    node_colors = ['#81c784' if G.nodes[n]['type'] == 'leaf' else '#64b5f6' for n in G.nodes()]

    # 4. Dibujar
    # Ajustar tamaño de figura según complejidad (ej: número de hojas)
    num_leaves = len([n for n in G.nodes() if G.nodes[n]['type'] == 'leaf'])
    plt.figure(figsize=(max(12, num_leaves * 1.5), 8))
    
    # Dibujar Nodos y Aristas
    nx.draw(G, pos, 
            with_labels=False, 
            node_color=node_colors, 
            node_size=3500, 
            edge_color='#bdbdbd', 
            width=1.5, 
            arrows=True, 
            arrowsize=20)
    
    # Dibujar Etiquetas de Nodos (con fondo blanco para legibilidad)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family='sans-serif')
    
    # Dibujar Etiquetas de Aristas (L/R)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='#e53935', font_weight='bold', font_size=11)

    plt.title("Estructura Jerárquica Binaria del Modelo Clasificador", fontsize=14, fontweight='bold', pad=20)
    plt.axis('off') # Ocultar ejes
    plt.tight_layout()
    
    # 5. Guardar
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Imagen del árbol guardada.")

# =========================================================
# 5. EJECUCIÓN DEL FLUJO SEGÚN JERARQUÍA
# =========================================================

if JERARQUIA == "TREE":
    print("\n" + "="*55)
    print(" 🌳 ENTRENAMIENTO ÁRBOL BINARIO ML (LOOCV EXACTO)")
    print("="*55)
    
    loo = LeaveOneOut()
    n = len(X_train)
    y_true_loo, y_pred_loo = [], []
    
    # LOOCV Exacto: Reentrenar el árbol completo por fold
    for train_idx, test_idx in tqdm(loo.split(X_train), total=n, desc="LOOCV Folds"):
        X_tr_fold = X_train.iloc[train_idx]
        y_sp_fold = y_train_sp.iloc[train_idx]
        X_te_fold = X_train.iloc[test_idx].values
        
        # Podar árbol a especies presentes en el fold
        tree_fold = construir_arbol_fijo(ARBOL_FIJO, especies_presentes=set(y_sp_fold.unique()))
        h_fold = entrenar_arbol_ml(X_tr_fold, y_sp_fold, tree_fold)
        
        pred_sp, _, _ = clasificar_cascada_ml(h_fold, X_te_fold)
        
        y_true_loo.append(y_train_sp.iloc[test_idx[0]])
        y_pred_loo.append(pred_sp[0])
        
    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    
    # Métricas LOOCV
    acc = accuracy_score(y_true_loo, y_pred_loo)
    bacc = balanced_accuracy_score(y_true_loo, y_pred_loo)
    print(f"\n🎯 Métricas LOOCV (Árbol):")
    print(f"Accuracy: {acc:.2%} | Balanced Accuracy: {bacc:.2%}")
    print(classification_report(y_true_loo, y_pred_loo))
    
    # --- Modelo Final con todos los datos ---
    tree_final = construir_arbol_fijo(ARBOL_FIJO, especies_presentes=set(y_train_sp.unique()))
    
    # 📸 NUEVO: GUARDAR IMAGEN DEL ÁRBOL FINAL
    guardar_imagen_arbol(tree_final, filename="estructura_arbol_final_ml.png")
    
    print("\nEntrenando modelos finales en cada nodo...")
    h_final = entrenar_arbol_ml(X_train, y_train_sp, tree_final)
    
    # Proyectar Fósiles
    if len(X_project) > 0:
        print("\n🦴 Proyectando Fósiles...")
        preds_sp, confs, sp_probs_list = clasificar_cascada_ml(h_final, X_project.values)
        all_extant_species = sorted(tree_final['members'])
        
        prob_rows = []
        final_predictions = []
        for i in range(len(X_project)):
            sp_id = y_project_ids.iloc[i]
            row = {'Specimen_Extinct': sp_id}
            sp_probs = sp_probs_list[i]
            for sp in all_extant_species:
                row[f"P_{sp.replace(' ', '_')}"] = np.round(sp_probs.get(sp, 0.0), 4)
            best_sp = preds_sp[i]
            row['Best_Extant_Specie'] = best_sp
            row['Best_Extant_Prob'] = np.round(sp_probs[best_sp], 4)
            prob_rows.append(row)
            
            final_predictions.append({
                'Specimen_Extinct': sp_id,
                'Pred_Specie': best_sp,
                'Prob_Conjunta': np.round(sp_probs[best_sp], 4)
            })

else:
    # Bloque original H3 (Original)
    print("Modo H3 (Original) seleccionado.")
    pass

# =========================================================
# 6. GRÁFICOS DE RESULTADOS Y EXPORTACIÓN
# =========================================================
if JERARQUIA == "TREE" and len(X_project) > 0:
    pd.DataFrame(final_predictions).to_csv("predicciones_arbol_ml.csv", index=False)
    pd.DataFrame(prob_rows).to_csv("probabilidades_arbol_ml.csv", index=False)
    print("📁 Guardados CSVs de predicciones y probabilidades.")

    # Matriz de Confusión LOOCV
    labels_sorted = np.unique(np.concatenate([y_true_loo, y_pred_loo]))
    cm = confusion_matrix(y_true_loo, y_pred_loo, labels=labels_sorted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.title(f"Matriz de Confusión LOOCV - Árbol ML\nBalanced Acc: {bacc:.2%}")
    plt.ylabel('Realidad')
    plt.xlabel('Predicción (LOO)')
    plt.tight_layout()
    plt.savefig("matriz_confusion_arbol_ml.png", dpi=300)
    plt.close()
    
    # Matriz de Elecciones (Extinct x Extant)
    extinct_species_labels = df_orig.loc[mask_extinct, 'Target_Specie'].reset_index(drop=True)
    count_matrix = pd.DataFrame(0, index=sorted(extinct_species_labels.unique()), columns=all_extant_species)
    for i, pred_info in enumerate(final_predictions):
        count_matrix.loc[extinct_species_labels.iloc[i], pred_info['Pred_Specie']] += 1

    plt.figure(figsize=(10, 6))
    sns.heatmap(count_matrix, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, linecolor='lightgrey')
    plt.title("Matriz de Elecciones Fósiles (Árbol ML)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("matriz_elecciones_extinct_extant_arbol_ml.png", dpi=200)
    plt.close()
    
    print("📊 Gráficos de resultados guardados correctamente.")