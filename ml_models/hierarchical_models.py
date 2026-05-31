import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# ⚙️  CONFIGURACIÓN DE CONJUNTOS (NIVEL 1) — JERARQUÍA BHC
#
#  Grupos definidos por el corte K=3 del dendrograma BHC
#  (UPGMA sobre distancias de Bhattacharyya, espacio latente 5D).
#  El BHC recuperó la filogenia conocida sin supervisión (NMI=0.85).
#
#  Orden de fusión en el dendrograma:
#    Papio anubis + P. hamadryas        (dist 0.20) ┐
#    + Lophocebus aterrimus             (dist 0.48) ├─► Grupo_Papionini_A
#    + Papio ursinus                    (dist 0.72) ┘
#    Cercocebus atys + Macaca sylvanus  (dist 0.80) ──► Grupo_Papionini_B
#    Theropithecus gelada               (dist 1.50) ──► Grupo_Theropithecus
#
#  Corte en K=3: respeta la división filogenética principal
#  y garantiza suficientes muestras por grupo para entrenar.
#  Diferencia clave respecto a la jerarquía manual anterior:
#  Lophocebus aterrimus pasa de Grupo B a Grupo A, porque el
#  BHC lo fusiona antes con el clado Papio (dist 0.48) que
#  con Cercocebus+Macaca (dist 0.80).
# =========================================================
SPECIES_SETS = {
    "Grupo_Papionini_A": [
        "Papio anubis",
        "Papio hamadryas",
        "Papio ursinus",
        "Lophocebus aterrimus",
    ],
    "Grupo_Papionini_B": [
        "Cercocebus atys",
        "Macaca sylvanus",
    ],
    "Grupo_Theropithecus": [
        "Theropithecus gelada",
    ],
}

# Construimos el mapa inverso: especie -> nombre de conjunto
SPECIES_TO_SET = {
    sp: set_name
    for set_name, species_list in SPECIES_SETS.items()
    for sp in species_list
}

# =========================================================
# 1. CARGA DE DATOS
#    - Nivel 1 (conjuntos) → espacio latente del GPLVM
#    - Nivel 2 (especies)  → features originales escaladas
# =========================================================
LATENT_PATH   = "src/espacio_latente_gplvm_1.csv"
ORIGINAL_PATH = "src/dataset_con_grupos.csv"

metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']

# --- CSV latente (Nivel 1) ---
try:
    df_latent = pd.read_csv(LATENT_PATH)
    print(f"✅ Datos latentes cargados:   {df_latent.shape}")
except FileNotFoundError as e:
    print(f"❌ Error al leer espacio latente: {e}")
    exit()

latent_feature_cols = [c for c in df_latent.columns if c.startswith('Latent_')]
X_latent_all = df_latent[latent_feature_cols]

# --- CSV original (Nivel 2) ---
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

# Ambos CSVs tienen el mismo orden de filas → máscaras del latente
status       = df_latent['Extant/Extinct '].astype(str)
mask_extant  = status.str.contains('Extant',  case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

y_specie_all = df_latent['Target_Specie']
y_set_all    = y_specie_all.map(SPECIES_TO_SET)

mask_train = mask_extant & y_set_all.notna()
ignored    = mask_extant & y_set_all.isna()
if ignored.sum() > 0:
    print(f"⚠️  {ignored.sum()} muestras extantes ignoradas (especie no asignada a ningún conjunto):")
    print(f"   {df_latent.loc[ignored, 'Target_Specie'].unique().tolist()}")

# Nivel 1 — espacio latente
X_train_L1   = X_latent_all[mask_train].reset_index(drop=True)
X_project_L1 = X_latent_all[mask_extinct].reset_index(drop=True)

# Nivel 2 — espacio original escalado
X_train_L2   = X_orig_scaled[mask_train].reset_index(drop=True)
X_project_L2 = X_orig_scaled[mask_extinct].reset_index(drop=True)

y_train_specie_raw = y_specie_all[mask_train].reset_index(drop=True)
y_train_set_raw    = y_set_all[mask_train].reset_index(drop=True)
y_project_ids      = df_latent.loc[mask_extinct, 'Specimen'].reset_index(drop=True)

# Codificamos el nivel 1 (conjuntos)
le_set = LabelEncoder()
y_train_set = le_set.fit_transform(y_train_set_raw)

print(f"\n🏷️  Conjuntos (Nivel 1 — espacio latente): {list(le_set.classes_)}")
for s, sps in SPECIES_SETS.items():
    print(f"   • {s}: {sps}")
print(f"\n🧬 Muestras de entrenamiento: {len(y_train_set)}")
print(f"🦴 Fósiles a proyectar: {len(y_project_ids)}")

# =========================================================
# 2. DEFINICIÓN DE MODELOS BASE
# =========================================================
base_models = {
    "Random Forest":      RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM":                SVC(class_weight='balanced', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
}

param_grids = {
    "Random Forest":      {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
    "SVM":                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10]},
}

loo = LeaveOneOut()

# =========================================================
# 3. NIVEL 1 — CLASIFICACIÓN DE CONJUNTO
# =========================================================
print("\n" + "="*55)
print(" FASE 1: CLASIFICACIÓN DE CONJUNTO (NIVEL 1)")
print("="*55)

cv_5_sets = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_summary_sets = {}
best_set_model_name = ""
best_set_acc = 0
final_set_models = {}

for name, model in base_models.items():
    print(f"⚙️  Optimizando {name} (Conjuntos)...")

    grid = GridSearchCV(model, param_grids[name], cv=cv_5_sets,
                        scoring='balanced_accuracy', n_jobs=-1)
    grid.fit(X_train_L1, y_train_set)

    best_est = grid.best_estimator_
    final_set_models[name] = best_est

    y_pred_loo = cross_val_predict(best_est, X_train_L1, y_train_set, cv=loo)
    bal_acc = balanced_accuracy_score(y_train_set, y_pred_loo)
    results_summary_sets[name] = bal_acc
    print(f"   Balanced Accuracy: {bal_acc:.2%} | Params: {grid.best_params_}")

    if bal_acc > best_set_acc:
        best_set_acc = bal_acc
        best_set_model_name = name

winner_set_model = final_set_models[best_set_model_name]
winner_set_model.fit(X_train_L1, y_train_set)
print(f"\n🥇 MEJOR MODELO NIVEL 1: {best_set_model_name} ({best_set_acc:.2%})")

# =========================================================
# 4. NIVEL 2 — UN MODELO EXPERTO POR CONJUNTO
# =========================================================
print("\n" + "="*55)
print(" FASE 2: MODELOS EXPERTOS POR CONJUNTO (NIVEL 2)")
print("="*55)

species_models   = {}   # set_name -> modelo entrenado (o "SINGLE_SPECIES")
species_encoders = {}   # set_name -> LabelEncoder de especies
species_cv_results = {} # set_name -> dict con y_true, y_pred, classes
species_loocv_dict = {} # índice global -> nombre de especie predicha (LOO)

for set_name, species_list in SPECIES_SETS.items():
    mask_g = (y_train_set_raw == set_name)
    X_train_g = X_train_L2[mask_g]
    y_train_sp_g_raw = y_train_specie_raw[mask_g]
    indices_g = np.where(mask_g.values)[0]

    le_sp = LabelEncoder()
    y_train_sp_g = le_sp.fit_transform(y_train_sp_g_raw)
    species_encoders[set_name] = le_sp
    num_species = len(le_sp.classes_)

    print(f"\n🔬 Conjunto '{set_name}': {num_species} especie(s) — {list(le_sp.classes_)}")

    if num_species == 1:
        print("   -> Solo 1 especie. Asignación directa.")
        species_models[set_name] = "SINGLE_SPECIES"
        for idx in indices_g:
            species_loocv_dict[idx] = le_sp.classes_[0]
        continue

    min_class = np.min(np.bincount(y_train_sp_g))
    if min_class >= 3:
        cv_expert = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    elif min_class == 2:
        cv_expert = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    else:
        cv_expert = KFold(n_splits=3, shuffle=True, random_state=42)

    best_expert_name  = ""
    best_expert_acc   = -1
    best_expert_model = None
    best_expert_preds = None

    for name, model in base_models.items():
        grid_sp = GridSearchCV(model, param_grids[name], cv=cv_expert,
                               scoring='balanced_accuracy', n_jobs=-1)
        grid_sp.fit(X_train_g, y_train_sp_g)
        best_est_sp = grid_sp.best_estimator_

        y_pred_sp_loo = cross_val_predict(best_est_sp, X_train_g, y_train_sp_g, cv=loo)
        bal_acc_sp = balanced_accuracy_score(y_train_sp_g, y_pred_sp_loo)

        print(f"   - {name}: {bal_acc_sp:.2%} | Params: {grid_sp.best_params_}")

        if bal_acc_sp > best_expert_acc:
            best_expert_acc   = bal_acc_sp
            best_expert_name  = name
            best_expert_model = best_est_sp
            best_expert_preds = y_pred_sp_loo

    print(f"   🏆 Ganador '{set_name}': {best_expert_name} ({best_expert_acc:.2%})")

    pred_sp_names = le_sp.inverse_transform(best_expert_preds)
    for i, idx in enumerate(indices_g):
        species_loocv_dict[idx] = pred_sp_names[i]

    species_cv_results[set_name] = {
        'y_true': y_train_sp_g,
        'y_pred': best_expert_preds,
        'classes': le_sp.classes_,
    }

    best_expert_model.fit(X_train_g, y_train_sp_g)
    species_models[set_name] = best_expert_model

# =========================================================
# 5. EVALUACIÓN GLOBAL EN CASCADA
# =========================================================
print("\n" + "="*55)
print(" FASE 3: PRECISIÓN FINAL EN CASCADA (ESPECIE)")
print("="*55)

global_species_acc  = {}
best_cascade_name   = ""
best_cascade_acc    = 0
best_cascade_preds  = []

for name, set_model in final_set_models.items():
    set_preds_loo   = cross_val_predict(set_model, X_train_L1, y_train_set, cv=loo)
    set_preds_names = le_set.inverse_transform(set_preds_loo)

    final_sp_preds = []

    for i in range(len(X_train_L1)):
        true_set = y_train_set_raw.iloc[i]
        pred_set = set_preds_names[i]

        if true_set == pred_set:
            # Nivel 1 correcto → usamos la predicción LOO del experto correcto
            final_sp_preds.append(species_loocv_dict[i])
        else:
            # Nivel 1 incorrecto → el experto equivocado hace la predicción
            sample       = X_train_L2.iloc[[i]]
            wrong_model   = species_models[pred_set]
            wrong_encoder = species_encoders[pred_set]

            if wrong_model == "SINGLE_SPECIES":
                final_sp_preds.append(wrong_encoder.classes_[0])
            else:
                sp_enc = wrong_model.predict(sample)
                final_sp_preds.append(wrong_encoder.inverse_transform(sp_enc)[0])

    bal_acc = balanced_accuracy_score(y_train_specie_raw, final_sp_preds)
    global_species_acc[name] = bal_acc
    print(f"   Accuracy cascada (base {name}): {bal_acc:.2%}")

    if bal_acc > best_cascade_acc:
        best_cascade_acc   = bal_acc
        best_cascade_name  = name
        best_cascade_preds = final_sp_preds

# =========================================================
# 6. PROYECCIÓN DE FÓSILES
# =========================================================
print("\n" + "="*55)
print(" FASE 4: PROYECTANDO FÓSILES")
print("="*55)

set_preds_enc   = winner_set_model.predict(X_project_L1)
set_preds_names = le_set.inverse_transform(set_preds_enc)
set_confidences = np.max(winner_set_model.predict_proba(X_project_L1), axis=1)

final_predictions = []

for i in range(len(X_project_L1)):
    sample        = X_project_L2.iloc[[i]]
    pred_set      = set_preds_names[i]
    expert_model  = species_models.get(pred_set)
    expert_enc    = species_encoders.get(pred_set)

    if expert_model is None:
        predicted_specie  = "Unknown"
        specie_confidence = 0.0
    elif expert_model == "SINGLE_SPECIES":
        predicted_specie  = expert_enc.classes_[0]
        specie_confidence = 1.00
    else:
        sp_enc            = expert_model.predict(sample)
        predicted_specie  = expert_enc.inverse_transform(sp_enc)[0]
        specie_confidence = np.max(expert_model.predict_proba(sample))

    final_predictions.append({
        'Specimen_Extinct': y_project_ids.iloc[i],
        'Pred_Conjunto':    pred_set,
        'Conf_Conjunto':    np.round(set_confidences[i], 2),
        'Pred_Specie':      predicted_specie,
        'Conf_Specie':      np.round(specie_confidence, 2),
    })

projection_table = pd.DataFrame(final_predictions)
output_csv = "predicciones_jerarquicas_conjuntos.csv"
projection_table.to_csv(output_csv, index=False)
print(f"📁 Guardado: {output_csv}")

# =========================================================
# 6b. PROBABILIDADES POR ESPECIE EXTANT PARA CADA FÓSIL
#     Calcula la probabilidad de cada especie extant para
#     cada fósil, combinando las dos fases del modelo
#     jerárquico: P(especie | fósil) = P(conjunto | fósil)
#                                    × P(especie | conjunto, fósil)
# =========================================================
print("\n" + "="*55)
print(" FASE 4b: PROBABILIDADES POR ESPECIE EXTANT (EXTINCT)")
print("="*55)

# Probabilidades de conjunto (Nivel 1) para los extinct
set_proba_matrix = winner_set_model.predict_proba(X_project_L1)  # shape: (n_extinct, n_conjuntos)
set_class_names  = le_set.inverse_transform(np.arange(len(le_set.classes_)))  # nombres de conjuntos

# Construimos la lista de todas las especies extant en orden canónico
all_extant_species = []
for set_name in le_set.classes_:
    le_sp = species_encoders[set_name]
    all_extant_species.extend(le_sp.classes_.tolist())

prob_rows = []

for i in range(len(X_project_L1)):
    sample_L2 = X_project_L2.iloc[[i]]
    row = {'Specimen_Extinct': y_project_ids.iloc[i]}

    # Para cada especie extant, P(especie|fósil) = Σ_conjunto P(conjunto|fósil) × P(especie|conjunto,fósil)
    species_probs = {sp: 0.0 for sp in all_extant_species}

    for j, set_name in enumerate(le_set.classes_):
        p_conjunto = set_proba_matrix[i, j]
        expert_model  = species_models[set_name]
        expert_encoder = species_encoders[set_name]

        if expert_model == "SINGLE_SPECIES":
            # Probabilidad 1.0 a la única especie de este conjunto
            sp = expert_encoder.classes_[0]
            species_probs[sp] += p_conjunto * 1.0
        else:
            sp_proba = expert_model.predict_proba(sample_L2)[0]  # shape: (n_especies_conjunto,)
            for k, sp in enumerate(expert_encoder.classes_):
                species_probs[sp] += p_conjunto * sp_proba[k]

    # Añadir probabilidades por especie a la fila
    for sp in all_extant_species:
        row[f"P_{sp.replace(' ', '_')}"] = np.round(species_probs[sp], 4)

    # Especie más probable y su probabilidad marginal
    best_sp   = max(species_probs, key=species_probs.get)
    best_prob = species_probs[best_sp]
    row['Best_Extant_Specie'] = best_sp
    row['Best_Extant_Prob']   = np.round(best_prob, 4)

    prob_rows.append(row)

probs_table = pd.DataFrame(prob_rows)
output_probs_csv = "predicciones_jerarquicas_probs_por_especie.csv"
probs_table.to_csv(output_probs_csv, index=False)
print(f"📁 Guardado: {output_probs_csv}")
print(probs_table[['Specimen_Extinct', 'Best_Extant_Specie', 'Best_Extant_Prob']].to_string(index=False))

# =========================================================
# 7. GRÁFICOS
# =========================================================
print("\n📊 Generando gráficos...")

# 7.1 Accuracy Nivel 1 (Conjuntos)
plt.figure(figsize=(9, 5))
bars = plt.bar(results_summary_sets.keys(), results_summary_sets.values(),
               color=['#264653', '#2a9d8f', '#e9c46a'])
plt.ylim(0, 1.15)
plt.title('Balanced Accuracy (LOOCV) — Nivel 1: Conjuntos')
plt.ylabel('Balanced Accuracy')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{bar.get_height():.1%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("accuracy_conjuntos.png", dpi=200)
plt.close()

# 7.2 Accuracy Final Cascada (Especies)
plt.figure(figsize=(9, 5))
bars_sp = plt.bar(global_species_acc.keys(), global_species_acc.values(),
                  color=['#1d3557', '#457b9d', '#e63946'])
plt.ylim(0, 1.15)
plt.title('Balanced Accuracy (LOOCV) — Cascada final hasta Especie')
plt.ylabel('Balanced Accuracy')
for bar in bars_sp:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{bar.get_height():.1%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("accuracy_cascada_especies.png", dpi=200)
plt.close()

# 7.3 Matriz de Confusión — Nivel 1 (Conjuntos)
y_pred_loo_winner = cross_val_predict(winner_set_model, X_train_L1, y_train_set, cv=loo)
cm_sets = confusion_matrix(y_train_set, y_pred_loo_winner)
cm_sets_norm = cm_sets.astype('float') / cm_sets.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_sets_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le_set.classes_, yticklabels=le_set.classes_)
plt.title(f'Matriz de Confusión: {best_set_model_name} (LOOCV)\nNivel 1 — Conjuntos')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.tight_layout()
plt.savefig("matriz_confusion_conjuntos.png", dpi=200)
plt.close()

# 7.4 Matrices de Confusión internas — Nivel 2 (Especies por Conjunto)
for set_name, res in species_cv_results.items():
    cm_sp = confusion_matrix(res['y_true'], res['y_pred'])
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_sp_norm = cm_sp.astype('float') / cm_sp.sum(axis=1)[:, np.newaxis]
        cm_sp_norm = np.nan_to_num(cm_sp_norm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sp_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=res['classes'], yticklabels=res['classes'])
    plt.title(f'Matriz de Confusión Interna (LOOCV)\nEspecies — {set_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.tight_layout()
    fname = f"matriz_confusion_especies_{set_name}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

# 7.5 Reporte textual como imagen
class_report_text = classification_report(y_train_specie_raw, best_cascade_preds)
texto_final = (
    f"=== MODELO JERÁRQUICO POR CONJUNTOS ===\n"
    f"Mejor modelo base (Conjuntos): {best_cascade_name}\n"
    f"Balanced Accuracy GLOBAL (Especies): {best_cascade_acc:.2%}\n\n"
    f"=== REPORTE DE CLASIFICACIÓN FINAL (ESPECIES) ===\n"
    f"{class_report_text}"
)
print("\n" + texto_final)

fig_text, ax_text = plt.subplots(figsize=(10, 8))
ax_text.axis('off')
ax_text.text(0.01, 0.99, texto_final, fontsize=11, fontfamily='monospace',
             va='top', ha='left', transform=ax_text.transAxes)
plt.savefig('metricas_resumen_jerarquico.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("-> Guardado: 'metricas_resumen_jerarquico.png'")

print("\n✅ Todo completado.")

# =========================================================
# 8. MATRIZ DE ELECCIONES: extinct × extant
#    Para cada especie extinct, cuántas veces fue elegida
#    cada especie extant como la predicción más probable.
# =========================================================
print("\n" + "="*55)
print(" FASE 5: MATRIZ DE ELECCIONES (EXTINCT × EXTANT)")
print("="*55)

# Recuperamos la especie de cada fósil
extinct_species_col = 'Target_Specie'
if extinct_species_col in df_latent.columns:
    extinct_species_labels = df_latent.loc[mask_extinct, extinct_species_col].reset_index(drop=True)
else:
    extinct_species_labels = y_project_ids  # fallback: usar Specimen como label

# Lista canónica de especies extant (en el mismo orden que el modelo)
all_extant_species_list = []
for set_name in le_set.classes_:
    le_sp = species_encoders[set_name]
    all_extant_species_list.extend(le_sp.classes_.tolist())

# Construimos la matriz de conteo
extinct_species_unique = extinct_species_labels.unique()
count_matrix = pd.DataFrame(
    0,
    index=sorted(extinct_species_unique),
    columns=all_extant_species_list
)

for i, pred_info in enumerate(final_predictions):
    extinct_sp = extinct_species_labels.iloc[i]
    chosen_extant = pred_info['Pred_Specie']
    if chosen_extant in count_matrix.columns:
        count_matrix.loc[extinct_sp, chosen_extant] += 1

# Guardamos como CSV
output_count_matrix_csv = "matriz_elecciones_extinct_extant.csv"
count_matrix.to_csv(output_count_matrix_csv)
print(f"📁 Guardado: {output_count_matrix_csv}")
print(count_matrix.to_string())

# Visualización como heatmap
plt.figure(figsize=(max(10, len(all_extant_species_list) * 1.2),
                    max(6,  len(extinct_species_unique) * 0.8)))
sns.heatmap(
    count_matrix,
    annot=True, fmt='d', cmap='YlOrRd',
    linewidths=0.5, linecolor='lightgrey',
    cbar_kws={'label': 'Nº de veces elegida'}
)
plt.title("Matriz de Elecciones: Especies Extinct × Especies Extant\n"
          "(Modelo Jerárquico — cuántas veces cada extant fue la predicción más probable)",
          fontsize=12)
plt.xlabel("Especie Extant elegida", fontsize=11)
plt.ylabel("Especie Extinct", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_elecciones_extinct_extant.png", dpi=200)
plt.close()
print("-> Guardado: 'matriz_elecciones_extinct_extant.png'")