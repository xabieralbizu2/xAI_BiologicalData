import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
TIPO_PRIOR = "lasso" # Opciones: "lasso", "gaussian", "horseshoe", "spike_and_slab"
FILE_PATH = "src/dataset_con_grupos.csv"
CHAINS = 4
DRAWS = 1000
TUNE = 1500

# =========================================================
# JERARQUÍA BHC (corte K=3, UPGMA sobre Bhattacharyya, ld5)
#
#  El BHC recuperó la filogenia sin supervisión (NMI=0.85).
#  Lophocebus aterrimus pasa al grupo Papio porque el BHC
#  lo fusiona con ese clado (dist 0.48) antes que con
#  Cercocebus+Macaca (dist 0.80). Ese es el único cambio
#  respecto a la jerarquía manual anterior.
#
#  Grupo_Papionini_A : Papio anubis, P. hamadryas,
#                      P. ursinus, Lophocebus aterrimus
#  Grupo_Papionini_B : Cercocebus atys, Macaca sylvanus
#  Grupo_Theropithecus: Theropithecus gelada
# =========================================================
BHC_GROUP_MAP = {
    "Papio anubis":          "Grupo_Papionini_A",
    "Papio hamadryas":       "Grupo_Papionini_A",
    "Papio ursinus":         "Grupo_Papionini_A",
    "Lophocebus aterrimus":  "Grupo_Papionini_A",
    "Cercocebus atys":       "Grupo_Papionini_B",
    "Macaca sylvanus":       "Grupo_Papionini_B",
    "Theropithecus gelada":  "Grupo_Theropithecus",
}

print(f"--- INICIANDO MODELO JERÁRQUICO BAYESIANO | PRIOR: {TIPO_PRIOR.upper()} ---")

# =========================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================================================
print("Cargando y filtrando datos...")
df = pd.read_csv(FILE_PATH)

metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']
feature_cols = [c for c in df.columns if c not in metadata_cols]

X_raw = df[feature_cols].fillna(0)
y_specie_all = df['Target_Specie']
status = df['Extant/Extinct '].astype(str)

# Aplicar jerarquía BHC: reemplaza Target_group por los grupos del dendrograma
y_group_all = df['Target_Specie'].map(BHC_GROUP_MAP).fillna(df['Target_group'])

# Estandarización global
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

mask_extant = status.str.contains('Extant', case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

# Sets de Entrenamiento (Vivos)
X_train = X_scaled[mask_extant]
y_train_group_raw = y_group_all[mask_extant]
y_train_specie_raw = y_specie_all[mask_extant]

# Sets de Proyección (Fósiles)
X_project = X_scaled[mask_extinct]
y_project_ids = df.loc[mask_extinct, 'Specimen']

le_group = LabelEncoder()
y_train_group = le_group.fit_transform(y_train_group_raw)

print(f"Grupos a predecir (Nivel 1): {len(le_group.classes_)}")
print(f"Muestras de entrenamiento (Extant): {len(X_train)}")
print(f"Muestras a proyectar (Extinct): {len(X_project)}")

# =========================================================
# 2. MOTOR BAYESIANO (FUNCIÓN REUTILIZABLE)
# =========================================================
def construir_y_muestrear_modelo(X_val, y_val, n_classes, tipo_prior):
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
        
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=0.99, random_seed=42, progressbar=True)
        
        # IMPORTANTE: Calcular log-likelihood para poder usar LOOCV después
        pm.compute_log_likelihood(idata)
        
    return model, idata

def predecir_bayesiano(model, idata, X_new):
    """Genera predicciones y confianza usando simulaciones a posteriori."""
    with model:
        pm.set_data({'X': X_new, 'y': np.zeros(len(X_new), dtype=int)})
        ppc = pm.sample_posterior_predictive(idata, predictions=True, var_names=['y_obs', 'p'], random_seed=42, progressbar=False)
        
    y_pred_samples = ppc.predictions['y_obs'].values
    y_pred_flat = y_pred_samples.reshape(-1, len(X_new))
    y_pred_consensus = stats.mode(y_pred_flat, axis=0, keepdims=True)[0].flatten()
    
    p_samples = ppc.predictions['p'].values
    p_mean = p_samples.reshape(-1, len(X_new), p_samples.shape[-1]).mean(axis=0)
    confianza = np.max(p_mean, axis=1)
    
    return y_pred_consensus, confianza

# =========================================================
# 3. ENTRENAMIENTO NIVEL 1: GRUPOS
# =========================================================
print("\n" + "="*50)
print(" FASE 1: ENTRENANDO MODELO GLOBAL (GRUPOS)")
print("="*50)
n_classes_group = len(le_group.classes_)
model_group, idata_group = construir_y_muestrear_modelo(X_train.values, y_train_group, n_classes_group, TIPO_PRIOR)

# =========================================================
# 4. ENTRENAMIENTO NIVEL 2: ESPECIES (SUB-MODELOS)
# =========================================================
print("\n" + "="*50)
print(" FASE 2: ENTRENANDO SUB-MODELOS EXPERTOS POR GRUPO")
print("="*50)

species_models = {}
species_encoders = {}
unique_groups = y_train_group_raw.unique()

for group in unique_groups:
    mask_g = (y_train_group_raw == group)
    X_train_g = X_train[mask_g].values
    y_train_specie_g_raw = y_train_specie_raw[mask_g]
    
    le_sp = LabelEncoder()
    y_train_specie_g = le_sp.fit_transform(y_train_specie_g_raw)
    species_encoders[group] = le_sp
    num_species = len(le_sp.classes_)
    
    print(f"\n🔬 Procesando Grupo '{group}': {num_species} especie(s).")
    
    if num_species == 1:
        print("   -> Solo 1 especie. Asignación directa.")
        species_models[group] = "SINGLE_SPECIES"
    else:
        print(f"   -> Entrenando modelo bayesiano para {num_species} especies...")
        mod_sp, idata_sp = construir_y_muestrear_modelo(X_train_g, y_train_specie_g, num_species, TIPO_PRIOR)
        species_models[group] = (mod_sp, idata_sp)

# =========================================================
# 5. EVALUACIÓN CASCADA Y MÉTRICAS
# =========================================================
print("\n" + "="*50)
print(" FASE 3: EVALUACIÓN EN CASCADA Y CREACIÓN DE IMÁGENES DE TEXTO")
print("="*50)

# --- A. Predicciones Nivel 1 (Grupos) ---
pred_group_idx, conf_group = predecir_bayesiano(model_group, idata_group, X_train.values)
pred_group_str = le_group.inverse_transform(pred_group_idx)

# --- B. Predicciones Nivel 2 (Especies en Cascada) ---
final_sp_preds = np.empty(len(X_train), dtype=object)

for group_str in unique_groups:
    idx_pred = np.where(pred_group_str == group_str)[0]
    if len(idx_pred) == 0:
        continue
    X_subset = X_train.iloc[idx_pred].values
    
    if species_models[group_str] == "SINGLE_SPECIES":
        final_sp_preds[idx_pred] = species_encoders[group_str].classes_[0]
    else:
        mod_sp, idata_sp = species_models[group_str]
        sp_pred_idx, _ = predecir_bayesiano(mod_sp, idata_sp, X_subset)
        final_sp_preds[idx_pred] = species_encoders[group_str].inverse_transform(sp_pred_idx)

# =========================================================
# 5.5 GENERACIÓN DE LAS IMÁGENES DE TEXTO (LOOCV + REPORTS)
# =========================================================
print("Generando imágenes de resumen (Métricas de Texto)...")

# --- IMAGEN 1: RESUMEN GRUPOS (LOOCV + REPORTE) ---
loo_group = az.loo(idata_group, pointwise=True)
report_group = classification_report(y_train_group_raw, pred_group_str)

texto_grupos = f"=== RESUMEN NUMÉRICO LOOCV (NIVEL 1: GRUPOS) ===\n"
texto_grupos += str(loo_group) + "\n\n"
texto_grupos += f"=== REPORTE DE CLASIFICACIÓN BAYESIANO (GRUPOS) ===\n"
texto_grupos += report_group

fig_text_g, ax_text_g = plt.subplots(figsize=(10, 8))
ax_text_g.axis('off')
ax_text_g.text(0.01, 0.99, texto_grupos, fontsize=11, fontfamily='monospace', va='top', ha='left', transform=ax_text_g.transAxes)
plt.savefig(f'metricas_texto_1_grupos_{TIPO_PRIOR}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"-> Guardado: 'metricas_texto_1_grupos_{TIPO_PRIOR}.png'")

# --- IMAGEN 2: RESUMEN GENERAL / ESPECIES (REPORTE GLOBAL) ---
report_general = classification_report(y_train_specie_raw, final_sp_preds)

texto_general = f"=== REPORTE DE CLASIFICACIÓN JERÁRQUICO FINAL (ESPECIES) ===\n"
texto_general += f"Nota: LOOCV no aplica al modelo global por ser un ensamble de modelos condicionales.\n\n"
texto_general += report_general

fig_text_sp, ax_text_sp = plt.subplots(figsize=(10, 8))
ax_text_sp.axis('off')
ax_text_sp.text(0.01, 0.99, texto_general, fontsize=11, fontfamily='monospace', va='top', ha='left', transform=ax_text_sp.transAxes)
plt.savefig(f'metricas_texto_2_especies_general_{TIPO_PRIOR}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"-> Guardado: 'metricas_texto_2_especies_general_{TIPO_PRIOR}.png'")


# Gráficos de Confusión Visuales (Opcional, pero recomendados)
print("Generando Heatmaps de Confusión...")
plt.figure(figsize=(10, 8))
cm_grupos = confusion_matrix(y_train_group_raw, pred_group_str, labels=le_group.classes_)
sns.heatmap(cm_grupos, annot=True, fmt='d', cmap='Blues', xticklabels=le_group.classes_, yticklabels=le_group.classes_)
plt.title(f'Matriz de Confusión Bayesiana - GRUPOS ({TIPO_PRIOR})')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.tight_layout()
plt.savefig(f"matriz_confusion_bayes_grupos_{TIPO_PRIOR}.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 10))
unique_species = np.unique(y_train_specie_raw)
cm_sp = confusion_matrix(y_train_specie_raw, final_sp_preds, labels=unique_species)
sns.heatmap(cm_sp, annot=True, fmt='d', cmap='Greens', xticklabels=unique_species, yticklabels=unique_species)
plt.title(f'Matriz de Confusión Bayesiana Jerárquica - ESPECIES ({TIPO_PRIOR})')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"matriz_confusion_bayes_jerarquico_especies_{TIPO_PRIOR}.png", dpi=300)
plt.close()

# =========================================================
# 6. PROYECCIÓN JERÁRQUICA (FÓSILES)
# =========================================================
print("\n" + "="*50)
print(" FASE 4: PROYECTANDO FÓSILES")
print("="*50)

if len(X_project) > 0:
    fossil_group_idx, fossil_group_conf = predecir_bayesiano(model_group, idata_group, X_project.values)
    fossil_group_str = le_group.inverse_transform(fossil_group_idx)
    
    fossil_predictions = []
    
    for i in range(len(X_project)):
        sample = X_project.iloc[[i]].values
        predicted_group = fossil_group_str[i]
        conf_g = fossil_group_conf[i]
        
        if species_models[predicted_group] == "SINGLE_SPECIES":
            predicted_specie = species_encoders[predicted_group].classes_[0]
            conf_sp = 1.00
        else:
            mod_sp, idata_sp = species_models[predicted_group]
            sp_pred_idx, conf_array = predecir_bayesiano(mod_sp, idata_sp, sample)
            predicted_specie = species_encoders[predicted_group].inverse_transform(sp_pred_idx)[0]
            conf_sp = conf_array[0]
            
        fossil_predictions.append({
            'Specimen_Extinct': y_project_ids.iloc[i],
            'Pred_Group': predicted_group,
            'Conf_Group': np.round(conf_g, 3),
            'Pred_Specie': predicted_specie,
            'Conf_Specie': np.round(conf_sp, 3)
        })

    projection_table = pd.DataFrame(fossil_predictions)
    output_csv = f"predicciones_fosiles_bayes_jerarquico_{TIPO_PRIOR}.csv"
    projection_table.to_csv(output_csv, index=False)
    print(f"📁 Fósiles procesados y guardados en: {output_csv}")

    # =========================================================
    # 6b. PROBABILIDADES POR ESPECIE EXTANT PARA CADA FÓSIL
    #     P(especie | fósil) = Σ_grupo P(grupo | fósil) × P(especie | grupo, fósil)
    #     Se combinan las probabilidades de ambos niveles
    #     para obtener la probabilidad marginal de cada
    #     especie extant sobre los datos extinct.
    # =========================================================
    print("\n" + "="*50)
    print(" FASE 4b: PROBABILIDADES POR ESPECIE EXTANT (EXTINCT)")
    print("="*50)

    def predecir_proba_bayesiano(model, idata, X_new):
        """Devuelve la probabilidad media a posteriori de cada clase."""
        with model:
            pm.set_data({'X': X_new, 'y': np.zeros(len(X_new), dtype=int)})
            ppc = pm.sample_posterior_predictive(
                idata, predictions=True, var_names=['p'],
                random_seed=42, progressbar=False
            )
        p_samples = ppc.predictions['p'].values          # (chains, draws, n, n_classes)
        p_mean = p_samples.reshape(-1, len(X_new), p_samples.shape[-1]).mean(axis=0)
        return p_mean  # shape: (n, n_classes)

    # Probabilidades de grupo (Nivel 1) para todos los fósiles a la vez
    group_proba_matrix = predecir_proba_bayesiano(
        model_group, idata_group, X_project.values
    )  # shape: (n_extinct, n_grupos)

    # Lista canónica de especies extant
    all_extant_species = []
    for group in unique_groups:
        le_sp = species_encoders[group]
        all_extant_species.extend(le_sp.classes_.tolist())

    prob_rows = []

    for i in range(len(X_project)):
        sample = X_project.iloc[[i]].values
        row = {'Specimen_Extinct': y_project_ids.iloc[i]}

        species_probs = {sp: 0.0 for sp in all_extant_species}

        for j, group_str in enumerate(le_group.classes_):
            p_grupo = group_proba_matrix[i, j]

            if species_models[group_str] == "SINGLE_SPECIES":
                sp = species_encoders[group_str].classes_[0]
                species_probs[sp] += p_grupo * 1.0
            else:
                mod_sp, idata_sp = species_models[group_str]
                sp_proba = predecir_proba_bayesiano(mod_sp, idata_sp, sample)[0]  # (n_especies,)
                for k, sp in enumerate(species_encoders[group_str].classes_):
                    species_probs[sp] += p_grupo * sp_proba[k]

        for sp in all_extant_species:
            row[f"P_{sp.replace(' ', '_')}"] = np.round(species_probs[sp], 4)

        best_sp   = max(species_probs, key=species_probs.get)
        best_prob = species_probs[best_sp]
        row['Best_Extant_Specie'] = best_sp
        row['Best_Extant_Prob']   = np.round(best_prob, 4)

        prob_rows.append(row)

    probs_table = pd.DataFrame(prob_rows)
    output_probs_csv = f"predicciones_fosiles_bayes_probs_por_especie_{TIPO_PRIOR}.csv"
    probs_table.to_csv(output_probs_csv, index=False)
    print(f"📁 Guardado: {output_probs_csv}")
    print(probs_table[['Specimen_Extinct', 'Best_Extant_Specie', 'Best_Extant_Prob']].to_string(index=False))
else:
    print("No hay fósiles para proyectar.")

print("\n✅ ¡Ejecución del modelo bayesiano jerárquico completada con éxito!")

# =========================================================
# FASE EXTRA: MATRIZ DE ELECCIONES (EXTINCT × EXTANT)
#    Para cada especie extinct, cuántas veces fue elegida
#    cada especie extant como la predicción más probable.
# =========================================================
if len(X_project) > 0:
    print("\n" + "="*50)
    print(" FASE EXTRA: MATRIZ DE ELECCIONES (EXTINCT × EXTANT)")
    print("="*50)

    # Etiqueta de especie para cada fósil
    extinct_species_col = 'Target_Specie'
    if extinct_species_col in df.columns:
        extinct_species_labels = df.loc[mask_extinct, extinct_species_col].reset_index(drop=True)
    else:
        extinct_species_labels = y_project_ids  # fallback

    # Lista canónica de especies extant
    all_extant_species_list = []
    for group in unique_groups:
        le_sp = species_encoders[group]
        all_extant_species_list.extend(le_sp.classes_.tolist())

    # Construimos la matriz de conteo
    extinct_species_unique = extinct_species_labels.unique()
    count_matrix = pd.DataFrame(
        0,
        index=sorted(extinct_species_unique),
        columns=all_extant_species_list
    )

    for i, pred_info in enumerate(fossil_predictions):
        extinct_sp = extinct_species_labels.iloc[i]
        chosen_extant = pred_info['Pred_Specie']
        if chosen_extant in count_matrix.columns:
            count_matrix.loc[extinct_sp, chosen_extant] += 1

    # Guardamos como CSV
    output_count_matrix_csv = f"matriz_elecciones_extinct_extant_bayes_h_{TIPO_PRIOR}.csv"
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
    plt.title(f"Matriz de Elecciones: Especies Extinct × Especies Extant\n"
              f"(BLR Jerárquico {TIPO_PRIOR.upper()} — predicción más probable por fósil)",
              fontsize=12)
    plt.xlabel("Especie Extant elegida", fontsize=11)
    plt.ylabel("Especie Extinct", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"matriz_elecciones_extinct_extant_bayes_h_{TIPO_PRIOR}.png", dpi=300)
    plt.close()
    print(f"-> Guardado: 'matriz_elecciones_extinct_extant_bayes_h_{TIPO_PRIOR}.png'")