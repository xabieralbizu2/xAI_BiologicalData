import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN DEL PRIOR (¡AQUÍ ELIGES EL MODELO!)
# =========================================================
# Opciones disponibles: "lasso", "horseshoe", "spike_and_slab"
TIPO_PRIOR = "lasso" 
print(f"--- INICIANDO SCRIPT CON PRIOR: {TIPO_PRIOR.upper()} ---")

# =========================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================================================
print("Cargando y filtrando datos...")
df = pd.read_csv('src/dataset_nbool_std.csv')

df['Extant/Extinct '] = df['Extant/Extinct '].str.strip()
extant_df = df[df['Extant/Extinct '] == 'Extant'].copy()

X_raw = extant_df.drop(columns=['Target_Specie', 'Extant/Extinct ', 'Specimen', 'ID'])
y_raw = extant_df['Target_Specie']

feature_names = X_raw.columns.tolist()

le = LabelEncoder()
y_idx = le.fit_transform(y_raw)
n_classes = len(le.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
n_features = X_scaled.shape[1]

print(f"Especies a clasificar: {n_classes}")
print(f"Variables predictoras: {n_features}")
print(f"Total de muestras (Extant): {len(y_idx)}")

# =========================================================
# 2. CONSTRUCCIÓN DEL MODELO Y MUESTREO (MCMC)
# =========================================================
print(f"\nConstruyendo y muestreando el modelo ({TIPO_PRIOR})...")
with pm.Model() as bayesian_logistic:
    
    X_data = pm.Data('X', X_scaled)
    y_data = pm.Data('y', y_idx)
    
    # Intercepto estándar (no penalizado fuertemente)
    alpha_rest = pm.Normal('alpha_rest', mu=0, sigma=5, shape=n_classes - 1)
    
    # -----------------------------------------------------
    # LÓGICA DE PRIORS INTERCAMBIABLES
    # -----------------------------------------------------
    shape_beta = (n_features, n_classes - 1)
    
    if TIPO_PRIOR == "lasso":
        # Prior de Laplace tradicional (L1 Bayesiano)
        beta_rest = pm.Laplace('beta_rest', mu=0, b=0.1, shape=shape_beta)

    elif TIPO_PRIOR == "gaussian":
        # Prior Normal tradicional (L2 / Ridge Bayesiano)
        # Sigma controla cuánta regularización aplicas (menor sigma = más penalización)
        beta_rest = pm.Normal('beta_rest', mu=0, sigma=1.0, shape=shape_beta)
        
    elif TIPO_PRIOR == "horseshoe":
        # Horseshoe (parametrización no centrada para evitar divergencias de NUTS)
        tau = pm.HalfCauchy('tau', beta=1) # Encogimiento global
        lam = pm.HalfCauchy('lam', beta=1, shape=shape_beta) # Encogimiento local
        z = pm.Normal('z', mu=0, sigma=1, shape=shape_beta) # Variable latente
        # beta = z * tau * lambda
        beta_rest = pm.Deterministic('beta_rest', z * tau * lam)
        
    elif TIPO_PRIOR == "spike_and_slab":
        # Relajación continua de Spike & Slab (para que NUTS pueda muestrear)
        # w = probabilidad de pertenecer a la Losa (Slab)
        w_local = pm.Beta('w_local', alpha=1, beta=1, shape=shape_beta)
        # Spike = muy concentrado en 0 (irrelevante)
        spike = pm.Normal('spike', mu=0, sigma=0.05, shape=shape_beta)
        # Slab = disperso (relevante)
        slab = pm.Normal('slab', mu=0, sigma=5.0, shape=shape_beta)
        # Combinación de ambos basada en el peso 'w'
        beta_rest = pm.Deterministic('beta_rest', w_local * slab + (1 - w_local) * spike)
        
    else:
        raise ValueError("TIPO_PRIOR no reconocido. Usa 'lasso', 'horseshoe' o 'spike_and_slab'.")
    # -----------------------------------------------------
    
    # Añadimos la clase base fijada a 0 para identificabilidad
    alpha = pt.concatenate([pt.zeros(1), alpha_rest])
    beta = pt.concatenate([pt.zeros((n_features, 1)), beta_rest], axis=1)
    
    # Ecuación logística multinomial
    mu = pm.math.dot(X_data, beta) + alpha
    p = pm.Deterministic('p', pm.math.softmax(mu, axis=1))
    
    y_obs = pm.Categorical('y_obs', p=p, observed=y_data)

    idata = pm.sample(
        draws=1000,         
        tune=1500,          
        chains=4,           
        target_accept=0.99, # Mantenemos alto para Horseshoe
        random_seed=42
    )
    
    print("\nGenerando simulaciones a posteriori para métricas y evaluación...")
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

# =========================================================
# 3. EXTRACCIÓN Y GRAFICACIÓN DE DISTRIBUCIONES (TODAS LAS VARIABLES)
# =========================================================
print("\nGenerando gráficos de distribuciones a posteriori para TODAS las variables...")

# Extraemos la distribución sin importar cómo la llamó el modelo internamente
beta_samples = idata.posterior['beta_rest'].values
beta_samples_flat = beta_samples.reshape(-1, n_features, n_classes - 1)

cols = 4
rows = int(np.ceil(n_features / cols))

fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
axes = axes.flatten() 
colors = sns.color_palette("husl", n_classes)

for idx in range(n_features):
    ax = axes[idx]
    
    # Especie Base (Fijada en 0)
    sns.kdeplot(x=np.zeros(beta_samples_flat.shape[0]), ax=ax, fill=False, color=colors[0], 
                linewidth=1.5, warn_singular=False)
    
    # Demás especies
    for c in range(n_classes - 1):
        sns.kdeplot(x=beta_samples_flat[:, idx, c], ax=ax, fill=False, color=colors[c+1], 
                    alpha=0.5, linewidth=1.5)
        
    ax.set_title(f"{feature_names[idx]}", fontsize=12)
    if idx >= n_features - cols:
        ax.set_xlabel("Coefficient", fontsize=10)
    if idx % cols == 0:
        ax.set_ylabel("Density", fontsize=10)

    ax.set_ylim(0, 5)
        
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)

for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

handles = [mpatches.Patch(color=colors[0], label=f"{le.classes_[0]} (Base)")]
for c in range(n_classes - 1):
    handles.append(mpatches.Patch(color=colors[c+1], alpha=0.5, label=f"{le.classes_[c+1]}"))

plt.tight_layout(rect=[0, 0.08, 1, 1]) 
fig.legend(handles=handles, title="Class", loc='lower center', ncol=n_classes, bbox_to_anchor=(0.5, 0))

plt.savefig(f'1_distribuciones_posteriori_todas_{TIPO_PRIOR}.png', dpi=300, bbox_inches='tight')
print(f"-> Guardado: '1_distribuciones_posteriori_todas_{TIPO_PRIOR}.png'")

# =========================================================
# 4. DIAGNÓSTICO LOOCV (MOTOR DEL MODELO)
# =========================================================
print("\nEvaluando estabilidad del modelo con PSIS-LOOCV...")

with bayesian_logistic:
    pm.compute_log_likelihood(idata)

loo_result = az.loo(idata, pointwise=True)
loo_text = str(loo_result) 
print("\nResumen Numérico LOOCV:")
print(loo_result)

plt.figure(figsize=(8, 5))
az.plot_khat(loo_result, show_bins=True)
plt.title(f"Diagnóstico de Pareto k - {TIPO_PRIOR.upper()}")
plt.axhline(0.7, color='red', linestyle='--', label='Límite de Peligro (0.7)')
plt.legend()
plt.tight_layout()
plt.savefig(f'2_loo_khat_plot_{TIPO_PRIOR}.png', dpi=300)
print(f"-> Guardado: '2_loo_khat_plot_{TIPO_PRIOR}.png'")

# =========================================================
# 5. RENDIMIENTO PREDICTIVO (MATRIZ DE CONFUSIÓN)
# =========================================================
print("\nCalculando precisión y matriz de confusión bayesiana...")

y_pred_samples = idata.posterior_predictive['y_obs'].values
y_pred_flat = y_pred_samples.reshape(-1, len(y_idx))

y_pred_consensus = stats.mode(y_pred_flat, axis=0, keepdims=True)[0].flatten()

class_report_text = classification_report(y_idx, y_pred_consensus, target_names=le.classes_)
print(f"\n--- REPORTE DE CLASIFICACIÓN BAYESIANO ({TIPO_PRIOR.upper()}) ---")
print(class_report_text)

cm = confusion_matrix(y_idx, y_pred_consensus)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Matriz de Confusión Bayesiana - {TIPO_PRIOR.upper()}', fontsize=14, fontweight='bold')
plt.xlabel('Especie Predicha por el Modelo', fontsize=12)
plt.ylabel('Especie Real (Verdad)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'3_matriz_confusion_{TIPO_PRIOR}.png', dpi=300)
print(f"-> Guardado: '3_matriz_confusion_{TIPO_PRIOR}.png'")

# =========================================================
# 6. GUARDAR MÉTRICAS DE TEXTO COMO IMAGEN
# =========================================================
print("\nGuardando los resultados de texto (LOOCV y Reporte) como imagen...")

texto_final = f"=== RESUMEN NUMÉRICO LOOCV ({TIPO_PRIOR.upper()}) ===\n"
texto_final += loo_text + "\n\n"
texto_final += f"=== REPORTE DE CLASIFICACIÓN BAYESIANO ({TIPO_PRIOR.upper()}) ===\n"
texto_final += class_report_text

fig_text, ax_text = plt.subplots(figsize=(10, 8))
ax_text.axis('off') 

ax_text.text(0.01, 0.99, texto_final, fontsize=12, fontfamily='monospace', 
             va='top', ha='left', transform=ax_text.transAxes)

plt.savefig(f'4_metricas_resumen_{TIPO_PRIOR}.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"-> Guardado: '4_metricas_resumen_{TIPO_PRIOR}.png'")

print("\n¡Ejecución completada con éxito! Revisa los 4 gráficos generados en tu carpeta.")


# =========================================================
# 7. PROBABILIDADES PREDICTIVAS (TODAS LAS VARIABLES)
# =========================================================
print("\nGenerando curvas de probabilidad para TODAS las variables (esto puede tardar unos segundos)...")

N_puntos = 100
x_grid = np.linspace(-3, 3, N_puntos)

# Configurar la cuadrícula de subplots
cols = 4
rows = int(np.ceil(n_features / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()
colors = sns.color_palette("husl", n_classes)

# Iteramos sobre cada variable
for idx, var_name in enumerate(feature_names):
    ax = axes[idx]
    
    # 1. Crear datos sintéticos: Variamos ESTA variable, el resto se queda en 0 (su media)
    X_sintetico = np.zeros((N_puntos, n_features))
    X_sintetico[:, idx] = x_grid
    
    # 2. Generar predicciones para esta configuración
    with bayesian_logistic:
        pm.set_data({'X': X_sintetico, 'y': np.zeros(N_puntos, dtype=int)})
        ppc_var = pm.sample_posterior_predictive(
            idata, 
            var_names=["p"], 
            random_seed=42,
            predictions=True,
            progressbar=False # Apagamos la barra para no saturar la consola
        )
        
    p_muestras = ppc_var.predictions['p'].values
    p_muestras_plano = p_muestras.reshape(-1, N_puntos, n_classes)
    p_media = p_muestras_plano.mean(axis=0)
    
    # 3. Graficar en su respectivo subplot
    for c in range(n_classes):
        ax.plot(x_grid, p_media[:, c], color=colors[c], linewidth=2)
        # Pasamos el parámetro 'ax=ax' para que ArviZ dibuje en el recuadro correcto
        az.plot_hdi(x_grid, p_muestras_plano[:, :, c], ax=ax, color=colors[c], fill_kwargs={'alpha': 0.15})
        
    ax.set_title(f"vs {var_name}", fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0) 
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Etiquetas de los ejes solo en los bordes para mantenerlo limpio
    if idx % cols == 0:
        ax.set_ylabel("Probabilidad", fontsize=10)
    if idx >= n_features - cols:
        ax.set_xlabel("Valor Estandarizado", fontsize=10)

# Borrar los recuadros sobrantes si n_features no es múltiplo de cols
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

# Crear una única leyenda global abajo
handles = [mpatches.Patch(color=colors[c], label=f"{le.classes_[c]}") for c in range(n_classes)]
fig.legend(handles=handles, title="Especie", loc='lower center', ncol=min(n_classes, 4), bbox_to_anchor=(0.5, 0))

# Ajustar márgenes para hacer hueco a la leyenda
plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig(f'5_curvas_prob_grid_{TIPO_PRIOR}.png', dpi=300, bbox_inches='tight')
print(f"-> Guardado: '5_curvas_prob_grid_{TIPO_PRIOR}.png'")

# =========================================================
# 8. PREDICCIÓN SOBRE DATOS EXTINCT
#    Después de entrenar con los extant, cargamos las filas
#    extinct, las pasamos por el modelo y guardamos:
#      · La especie extant más probable para cada fósil
#      · La probabilidad de cada especie extant (ranking)
#    Nota: la predicción NO es la especie real del fósil
#    (que es desconocida o distinta), sino la especie
#    extant a la que más se parece morfológicamente.
# =========================================================
print("\n" + "="*50)
print(" FASE EXTRA: PREDICCIÓN SOBRE DATOS EXTINCT")
print("="*50)

df_full = pd.read_csv('src/dataset_nbool_std.csv')
df_full['Extant/Extinct '] = df_full['Extant/Extinct '].str.strip()
extinct_df = df_full[df_full['Extant/Extinct '] == 'Extinct'].copy()

if len(extinct_df) == 0:
    print("No se encontraron filas con 'Extinct' en el dataset.")
else:
    X_extinct_raw = extinct_df.drop(columns=['Target_Specie', 'Extant/Extinct ', 'Specimen', 'ID'],
                                     errors='ignore')
    # Aseguramos las mismas columnas que el entrenamiento
    X_extinct_raw = X_extinct_raw.reindex(columns=feature_names, fill_value=0)
    X_extinct_scaled = scaler.transform(X_extinct_raw)

    extinct_ids = extinct_df['Specimen'].values

    print(f"Muestras extinct a predecir: {len(extinct_ids)}")
    print("Calculando probabilidades a posteriori para los fósiles...")

    with bayesian_logistic:
        pm.set_data({'X': X_extinct_scaled, 'y': np.zeros(len(X_extinct_scaled), dtype=int)})
        ppc_extinct = pm.sample_posterior_predictive(
            idata, predictions=True, var_names=['p'],
            random_seed=42, progressbar=True
        )

    # p_mean shape: (n_extinct, n_classes)
    p_samples_ext = ppc_extinct.predictions['p'].values
    p_mean_ext = p_samples_ext.reshape(-1, len(X_extinct_scaled), n_classes).mean(axis=0)

    # Especie más probable para cada fósil
    best_idx_ext  = np.argmax(p_mean_ext, axis=1)
    best_sp_ext   = le.inverse_transform(best_idx_ext)
    best_prob_ext = np.max(p_mean_ext, axis=1)

    # Tabla completa con probabilidades por especie
    prob_rows_ext = []
    for i, specimen in enumerate(extinct_ids):
        row = {'Specimen_Extinct': specimen}
        for c, sp_name in enumerate(le.classes_):
            row[f"P_{sp_name.replace(' ', '_')}"] = np.round(p_mean_ext[i, c], 4)
        row['Best_Extant_Specie'] = best_sp_ext[i]
        row['Best_Extant_Prob']   = np.round(best_prob_ext[i], 4)
        prob_rows_ext.append(row)

    probs_extinct_table = pd.DataFrame(prob_rows_ext)
    output_extinct_csv = f"predicciones_extinct_blr_{TIPO_PRIOR}.csv"
    probs_extinct_table.to_csv(output_extinct_csv, index=False)
    print(f"-> Guardado: '{output_extinct_csv}'")
    print(probs_extinct_table[['Specimen_Extinct', 'Best_Extant_Specie', 'Best_Extant_Prob']].to_string(index=False))

    # Gráfico: distribución de las asignaciones (¿a qué especie extant se parecen más los fósiles?)
    assignment_counts = pd.Series(best_sp_ext).value_counts()
    plt.figure(figsize=(10, 5))
    bars = plt.bar(assignment_counts.index, assignment_counts.values,
                   color=sns.color_palette("husl", len(assignment_counts)))
    plt.title(f"Asignación de Fósiles a Especies Extant — BLR ({TIPO_PRIOR.upper()})")
    plt.xlabel("Especie Extant más probable")
    plt.ylabel("Número de Fósiles asignados")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(int(bar.get_height())), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'6_asignacion_fosiles_extant_{TIPO_PRIOR}.png', dpi=300)
    plt.close()
    print(f"-> Guardado: '6_asignacion_fosiles_extant_{TIPO_PRIOR}.png'")

    # =========================================================
    # MATRIZ DE ELECCIONES: extinct × extant
    #    Para cada especie extinct, cuántas veces fue elegida
    #    cada especie extant como la predicción más probable.
    # =========================================================
    print("\n" + "="*50)
    print(" FASE EXTRA 2: MATRIZ DE ELECCIONES (EXTINCT × EXTANT)")
    print("="*50)

    # Etiqueta de especie para cada fósil (si existe en el CSV)
    extinct_species_col = 'Target_Specie'
    if extinct_species_col in extinct_df.columns:
        extinct_species_labels = extinct_df[extinct_species_col].values
    else:
        extinct_species_labels = extinct_ids  # fallback: Specimen

    extinct_species_unique = pd.unique(extinct_species_labels)
    extant_species_list = list(le.classes_)  # orden canónico del LabelEncoder

    count_matrix = pd.DataFrame(
        0,
        index=sorted(extinct_species_unique),
        columns=extant_species_list
    )

    for i in range(len(extinct_ids)):
        extinct_sp = extinct_species_labels[i]
        chosen_extant = best_sp_ext[i]
        if chosen_extant in count_matrix.columns:
            count_matrix.loc[extinct_sp, chosen_extant] += 1

    # Guardamos como CSV
    output_count_matrix_csv = f"7_matriz_elecciones_extinct_extant_{TIPO_PRIOR}.csv"
    count_matrix.to_csv(output_count_matrix_csv)
    print(f"📁 Guardado: '{output_count_matrix_csv}'")
    print(count_matrix.to_string())

    # Visualización como heatmap
    plt.figure(figsize=(max(10, len(extant_species_list) * 1.2),
                        max(6,  len(extinct_species_unique) * 0.8)))
    sns.heatmap(
        count_matrix,
        annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, linecolor='lightgrey',
        cbar_kws={'label': 'Nº de veces elegida'}
    )
    plt.title(f"Matriz de Elecciones: Especies Extinct × Especies Extant\n"
              f"(BLR {TIPO_PRIOR.upper()} — predicción más probable por fósil)",
              fontsize=12)
    plt.xlabel("Especie Extant elegida", fontsize=11)
    plt.ylabel("Especie Extinct", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"7_matriz_elecciones_extinct_extant_{TIPO_PRIOR}.png", dpi=300)
    plt.close()
    print(f"-> Guardado: '7_matriz_elecciones_extinct_extant_{TIPO_PRIOR}.png'")

print("\n¡Ejecución completada con éxito! Revisa todos los archivos generados.")