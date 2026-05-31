import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  

# Herramientas de Scikit-Learn
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning 
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración de Warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. CARGA DE DATOS Y TARGET ÚNICO (ESPECIE)
# =========================================================
file_path = "src/dataset_con_grupos.csv" # Tu archivo

try:
    df = pd.read_csv(file_path)
    print(f"✅ Datos cargados: {df.shape}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

metadata_cols = ['Target_Specie', 'Target_group', 'Extant/Extinct ', 'Specimen', 'ID']
feature_cols = [c for c in df.columns if c not in metadata_cols]

# Separar Features y rellenar nulos
X_raw = df[feature_cols].fillna(0)
y_all = df['Target_Specie'] # TARGET DIRECTO: ESPECIE
status = df['Extant/Extinct '].astype(str)

# ESTANDARIZAMOS LAS FEATURES
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

mask_extant = status.str.contains('Extant', case=False, na=False)
mask_extinct = status.str.contains('Extinct', case=False, na=False)

# Sets de entrenamiento (Vivos)
X_train = X_scaled[mask_extant]
y_train_raw = y_all[mask_extant]

# Sets de proyección (Fósiles)
X_project = X_scaled[mask_extinct]
y_project_ids = df.loc[mask_extinct, 'Specimen']

# Codificamos los nombres de las ESPECIES a números
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

print(f"🏷️ Especies a predecir directamente: {len(le.classes_)}")
print(f"🧬 Especímenes de entrenamiento (Vivos): {len(y_train)}")
print(f"🦴 Fósiles a proyectar: {len(y_project_ids)}")

# =========================================================
# 2. DEFINICIÓN DE MODELOS (CON PROTECCIÓN DE DESBALANCEO)
# =========================================================
base_models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
}

param_grids = {
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
    "SVM": {'C': [5, 10, 15], 'kernel': ['linear', 'rbf']},
    "Logistic Regression": {'C': [0.05, 0.1, 0.3, 0.5]}
}

loo = LeaveOneOut()

# Lógica dinámica para la validación cruzada (GridSearch)
min_class_samples = np.min(np.bincount(y_train))
if min_class_samples >= 5:
    cv_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
elif min_class_samples >= 3:
    cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
elif min_class_samples == 2:
    cv_search = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
else:
    cv_search = KFold(n_splits=3, shuffle=True, random_state=42)

# =========================================================
# 3. OPTIMIZACIÓN Y EVALUACIÓN DIRECTA (LOOCV)
# =========================================================
print("\n" + "="*50)
print(" BUSCANDO EL MEJOR MODELO (CLASIFICACIÓN DIRECTA)")
print("="*50)

results_summary = {}
best_model_name = ""
best_acc = 0
final_models = {}

for name, model in base_models.items():
    print(f"⚙️ Optimizando {name}...")
    
    grid = GridSearchCV(model, param_grids[name], cv=cv_search, scoring='balanced_accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    final_models[name] = best_estimator 
    
    y_pred_loocv = cross_val_predict(best_estimator, X_train, y_train, cv=loo)
    
    bal_acc = balanced_accuracy_score(y_train, y_pred_loocv)
    results_summary[name] = bal_acc
    
    print(f"   Balanced Accuracy: {bal_acc:.2%} | Params: {grid.best_params_}")

    if bal_acc > best_acc:
        best_acc = bal_acc
        best_model_name = name

print(f"\n🥇 EL MEJOR MODELO DIRECTO ES: {best_model_name} con {best_acc:.2%} de Balanced Accuracy")

# =========================================================
# 4. ENTRENAMIENTO FINAL Y PROYECCIÓN DE FÓSILES
# =========================================================
print("\n" + "="*50)
print(" PROYECTANDO FÓSILES")
print("="*50)

winner_model = final_models[best_model_name]
winner_model.fit(X_train, y_train) # Re-entrenamos con todo el dataset

preds_encoded = winner_model.predict(X_project)
probs = winner_model.predict_proba(X_project)
confidence = np.max(probs, axis=1)

projection_table = pd.DataFrame()
projection_table['Specimen_Extinct'] = y_project_ids.values
projection_table['Pred_Specie'] = le.inverse_transform(preds_encoded)
projection_table['Conf_Specie'] = np.round(confidence, 2)

output_csv = "predicciones_directas_especies.csv"
projection_table.to_csv(output_csv, index=False)
print(f"📁 Fósiles procesados y guardados en: {output_csv}")

# =========================================================
# 5. GRÁFICOS
# =========================================================
print("\n📊 Generando gráficos de rendimiento...")

# Calculamos las predicciones LOOCV del modelo ganador para usar en los gráficos y el reporte
y_pred_loocv_winner = cross_val_predict(winner_model, X_train, y_train, cv=loo)

# GRÁFICO 1: Comparativa de Balanced Accuracy
plt.figure(figsize=(9, 5))
names = list(results_summary.keys())
values = list(results_summary.values())
bars = plt.bar(names, values, color=['#6a4c93', '#1982c4', '#8ac926'])
plt.ylim(0, 1.15)
plt.title('Balanced Accuracy (LOOCV) - Clasificación Directa de Especies')
plt.ylabel('Balanced Accuracy')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f"{bar.get_height():.1%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("accuracy_comparison_directo_especies.png")
plt.close()
print("-> Guardado: 'accuracy_comparison_directo_especies.png'")

# GRÁFICO 2: Matriz de Confusión del GANADOR
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_train, y_pred_loocv_winner)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=le.classes_, yticklabels=le.classes_,
            annot_kws={"size": 8})
plt.title(f'Matriz de Confusión: {best_model_name} (LOOCV)\nObjetivo: ESPECIES (Directo)') 
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_confusion_directo_especies.png")
plt.close()
print("-> Guardado: 'matriz_confusion_directo_especies.png'")


# =========================================================
# 6. GUARDAR MÉTRICAS DE TEXTO COMO IMAGEN (NUEVO)
# =========================================================
print("\n📝 Guardando el reporte de clasificación como imagen...")

# Extraemos el texto del reporte de sklearn
class_report_text = classification_report(y_train, y_pred_loocv_winner, target_names=le.classes_)

# Construimos el texto completo
texto_final = f"=== RESUMEN DE MODELO GANADOR (LOOCV) ===\n"
texto_final += f"Algoritmo: {best_model_name}\n"
texto_final += f"Balanced Accuracy general: {best_acc:.2%}\n\n"
texto_final += "=== REPORTE DE CLASIFICACIÓN ===\n"
texto_final += class_report_text

print(texto_final) # Lo imprimimos en consola también

# Creamos una figura en blanco para dibujar el texto
fig_text, ax_text = plt.subplots(figsize=(10, 8))
ax_text.axis('off') # Ocultamos los ejes

# Escribimos el texto usando una fuente monoespaciada para mantener las columnas alineadas
ax_text.text(0.01, 0.99, texto_final, fontsize=12, fontfamily='monospace', 
             va='top', ha='left', transform=ax_text.transAxes)

plt.savefig('metricas_resumen_ml.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("-> Guardado: 'metricas_resumen_ml.png'")

print("\n✅ Todo completado con éxito. Ejecución finalizada.")