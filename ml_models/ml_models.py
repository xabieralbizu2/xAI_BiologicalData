import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import xgboost as xgb
import os

# Parametros por defecto
try:
    from src.model_params import (
        param_grid_dt, param_grid_logreg, param_grid_mlp, 
        param_grid_nb, param_grid_lda, param_grid_xgb
    )
except ImportError:
    print("⚠️ No se encontró 'src.model_params'. Usando grids por defecto.")
    param_grid_dt = {'max_depth': [3, 5, 10], 'criterion': ['gini', 'entropy']}
    param_grid_logreg = {'C': [0.1, 1, 10]}
    param_grid_mlp = {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
    param_grid_nb = {'var_smoothing': [1e-9, 1e-8]}
    param_grid_lda = {'solver': ['svd', 'lsqr']}
    param_grid_xgb = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}

# ---------------------------------------------------------
# 1. LISTAS DE VARIABLES (Nombres EXACTOS)
# ---------------------------------------------------------
ISO_VARS = [
    'Sq', 'Ssk', 'Sku', 'Sp', 'Sv', 'Sz', 'Sa', 
    'Sal (s = 0.2)', 'Std (Reference angle = 0°)', 'Sdq', 'Sdr', 
    'Smr (c = 1 µm below highest peak)', 'Smc (p = 10%)', 'Sdc (p = 10%, q = 90%)', 
    'Vm (p = 10%)', 'Vv (p = 10%)', 'Vmp (p = 10%)', 'Vmc (p = 10%, q = 80%)', 
    'Vvc (p = 10%, q = 80%)', 'Vvv (p = 80%)', 'Sk', 'Spk', 'Svk', 
    'Smrk1', 'Smrk2', 'Sak1', 'Sak2', 'Spkx', 'Svkx', 
    'Spd (Pruning = 5% of Sz)', 'Spc (Pruning = 5% of Sz)', 'Sda (Pruning = 5% of Sz)', 
    'Sha (Pruning = 5% of Sz)', 'Sdv (Pruning = 5% of Sz)', 'Shv (Pruning = 5% of Sz)', 
    'Svd (Pruning = 5% of Sz)', 'Svc (Pruning = 5% of Sz)', 'Shh (Pruning = 5% of Sz)', 
    'Shhx (Pruning = 5% of Sz)', 'Shhq (Pruning = 5% of Sz)', 'Shax (Pruning = 5% of Sz)', 
    'Shaq (Pruning = 5% of Sz)', 'Shvx (Pruning = 5% of Sz)', 'Shvq (Pruning = 5% of Sz)', 
    'Sdd (Pruning = 5% of Sz)', 'Sddx (Pruning = 5% of Sz)', 'Sddq (Pruning = 5% of Sz)', 
    'Sdax (Pruning = 5% of Sz)', 'Sdaq (Pruning = 5% of Sz)', 'Sdvx (Pruning = 5% of Sz)', 
    'Sdvq (Pruning = 5% of Sz)', 'Shn (Pruning = 5% of Sz)', 'Sdn (Pruning = 5% of Sz)', 
    'Shrn (Pruning = 5% of Sz)', 'Shrnx (Pruning = 5% of Sz)', 'Shrnq (Pruning = 5% of Sz)', 
    'Shff (Pruning = 5% of Sz)', 'Shffx (Pruning = 5% of Sz)', 'Shffq (Pruning = 5% of Sz)', 
    'Shed (Pruning = 5% of Sz)', 'Shedx (Pruning = 5% of Sz)', 'Shedq (Pruning = 5% of Sz)', 
    'Shar (Pruning = 5% of Sz)', 'Sharx (Pruning = 5% of Sz)', 'Sharq (Pruning = 5% of Sz)', 
    'Sdrn (Pruning = 5% of Sz)', 'Sdrnx (Pruning = 5% of Sz)', 'Sdrnq (Pruning = 5% of Sz)', 
    'Sdff (Pruning = 5% of Sz)', 'Sdffx (Pruning = 5% of Sz)', 'Sdffq (Pruning = 5% of Sz)', 
    'Sded (Pruning = 5% of Sz)', 'Sdedx (Pruning = 5% of Sz)', 'Sdedq (Pruning = 5% of Sz)', 
    'Sdar (Pruning = 5% of Sz)', 'Sdarx (Pruning = 5% of Sz)', 'Sdarq (Pruning = 5% of Sz)', 
    'FLTt', 'FLTp', 'FLTv', 'FLTq'
]

FURROW_VARS = [
    'Maximum depth of furrows', 'Mean depth of furrows', 'Mean density of furrows'
]

SSFA_VARS = [
    'Fractal complexity (Asfc)', 'HAsfc81 (HAsfc81)', 'Median of Asfc (MedianAsfc)', 
    'Mean of Asfc (MeanAsfc)', 'Length-scale anisotropy (Sfrax) (epLsar)', 
    'Length-scale anisotropy (NewEplsar)'
]

ALL_NUMERIC_VARS = ISO_VARS + FURROW_VARS + SSFA_VARS

# ---------------------------------------------------------
# 2. FUNCIONES DE PREPROCESAMIENTO
# ---------------------------------------------------------
def read_dataset(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path, header=0)
    return pd.read_excel(path, header=1)

def transform_to_valid_cols(df: pd.DataFrame) -> pd.DataFrame:
    valid_cols = [c for c in ALL_NUMERIC_VARS if c in df.columns]
    # Importante: rellenar nulos
    df_numeric = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_numeric

def transform_under_corr(df: pd.DataFrame, threshold: float = 0.95, visualization: bool = False) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if visualization:
        print(f"Reducción de correlación: Eliminando {len(to_drop)} variables redundantes.")
        
    return df.drop(columns=to_drop)

# ---------------------------------------------------------
# 3. FUNCIONES DE ENTRENAMIENTO Y REGISTRO
# ---------------------------------------------------------
def run_model_grid(estimator, param_grid, X_train, y_train, name):
    print(f"\n--- Entrenando {name} (GridSearch) ---")
    
    # Usamos StratifiedKFold si es posible, o LOOCV si son muy pocos datos
    # Como filtramos solo extinct, asumimos pocos datos -> LOOCV es más seguro
    cv_strategy = LeaveOneOut()
    
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_strategy, 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejor Accuracy Validación ({name}): {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

def run_xgboost(X_train, y_train, le):
    print("\n--- Entrenando XGBoost ---")
    y_train_encoded = le.transform(y_train)
    
    xgb_base = xgb.XGBClassifier(
        n_estimators=100, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid_xgb,
        scoring='accuracy',
        cv=LeaveOneOut(),
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train_encoded)
    print(f"Mejor Accuracy Validación (XGBoost): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

def evaluate_and_log(results_list, model_name, model, X_test, y_test, cv_score, best_params, label_encoder=None):
    """
    Evalúa el modelo en el set de TEST (Extinct reservado) y guarda todas las métricas.
    """
    # Predicción
    y_pred = model.predict(X_test)
    
    # Métricas para XGBoost (numéricas) vs Sklearn (strings)
    if label_encoder is not None:
        y_test_numeric = label_encoder.transform(y_test)
        # y_pred ya es numérico en xgb
        acc = accuracy_score(y_test_numeric, y_pred)
        f1 = f1_score(y_test_numeric, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test_numeric, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_numeric, y_pred, average='weighted', zero_division=0)
    else:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    result_entry = {
        "Modelo": model_name,
        "CV_Accuracy_Train": round(cv_score, 4),    # Rendimiento en validación interna (sobre el 80% Train)
        "Test_Accuracy": round(acc, 4),             # Rendimiento en datos no vistos (el 20% Test)
        "Test_F1_Weighted": round(f1, 4),
        "Test_Precision": round(prec, 4),
        "Test_Recall": round(rec, 4),
        "Best_Params": str(best_params)
    }
    
    results_list.append(result_entry)
    print(f" -> Resultados guardados para {model_name}")

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    results_log = []

    # --- A. Carga ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE = os.path.join(BASE_DIR, "src/dataset_complete_onehot_norm.csv")

    df = read_dataset(FILE)

    # --- B. Limpieza ---
    df.columns = df.columns.str.strip()
    STATUS_COL = "Extant/Extinct"
    STATUS_COL2 = "Status"
    TARGET_COL = "Specie"

    if STATUS_COL in df.columns:
        df[STATUS_COL] = df[STATUS_COL].astype(str).str.strip()
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
        # --- C. FILTRADO (SOLO EXTINCT) ---
        # Aquí está el cambio clave: Nos quedamos SOLO con los Extinct
        df_extinct = df[df[STATUS_COL] == 'Extinct'].copy()

    elif STATUS_COL2 in df.columns:
        df[STATUS_COL2] = df[STATUS_COL2].astype(str).str.strip()
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
        # --- C. FILTRADO (SOLO EXTINCT) ---
        # Aquí está el cambio clave: Nos quedamos SOLO con los Extinct
        df_extinct = df[df[STATUS_COL2] == 'Extinct'].copy()

    else:
        print("❌ Error: Columna de estado no encontrada.")
        exit()

    # --- C. FILTRADO (SOLO EXTINCT) ---
    # Aquí está el cambio clave: Nos quedamos SOLO con los Extinct

    

    # --- D. División Train / Test (80/20) sobre Extinct ---
    X_full = transform_to_valid_cols(df_extinct)
    y_full = df_extinct[TARGET_COL]

    # Intentamos división estratificada, si falla (clases con 1 muestra), hacemos aleatoria
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        print("División Train/Test: Estratificada (balanceada por especie).")
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )
        print("⚠️ Aviso: División Train/Test aleatoria simple (algunas especies tenían muy pocos datos).")

    print(f"Shape Train: {X_train.shape} | Shape Test: {X_test.shape}")

    # --- E. Preprocesamiento (Fit en Train, Transform en Test) ---
    print("\n>>> PREPROCESAMIENTO")
    
    # 1. Reducción de Correlación
    # Aprendemos qué columnas borrar usando solo Train
    X_train_reduced = transform_under_corr(X_train, threshold=0.95, visualization=True)
    cols_kept = X_train_reduced.columns
    X_test_reduced = X_test[cols_kept]

    # 2. Estandarización
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(scaler.fit_transform(X_train_reduced), columns=cols_kept, index=X_train_reduced.index)
    X_test_std = pd.DataFrame(scaler.transform(X_test_reduced), columns=cols_kept, index=X_test_reduced.index)

    # 3. Normalización (Yeo-Johnson)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_train_yj = pd.DataFrame(pt.fit_transform(X_train_reduced), columns=cols_kept, index=X_train_reduced.index)
    X_test_yj = pd.DataFrame(pt.transform(X_test_reduced), columns=cols_kept, index=X_test_reduced.index)

    # --- F. Ejecución de Modelos ---
    
    # 1. Decision Tree
    model, cv_score, params = run_model_grid(DecisionTreeClassifier(random_state=42), param_grid_dt, X_train_reduced, y_train, "Decision Tree")
    evaluate_and_log(results_log, "Decision Tree", model, X_test_reduced, y_test, cv_score, params)

    # 2. XGBoost
    le = LabelEncoder()
    # Ajustamos encoder con todas las etiquetas posibles en train (cuidado si hay etiquetas nuevas en test, fallará)
    # Una buena práctica es ajustar el encoder con y_full para conocer todas las clases, aunque solo entrenemos con y_train
    le.fit(y_full) 
    try:
        model, cv_score, params = run_xgboost(X_train_reduced, y_train, le)
        evaluate_and_log(results_log, "XGBoost", model, X_test_reduced, y_test, cv_score, params, label_encoder=le)
    except Exception as e:
        print(f"⚠️ Error en XGBoost (probablemente clases faltantes en train): {e}")

    # 3. Logistic Regression
    model, cv_score, params = run_model_grid(LogisticRegression(max_iter=5000, random_state=42), param_grid_logreg, X_train_std, y_train, "Logistic Regression")
    evaluate_and_log(results_log, "Logistic Regression", model, X_test_std, y_test, cv_score, params)

    # 4. MLP Classifier
    model, cv_score, params = run_model_grid(MLPClassifier(max_iter=3000, random_state=42), param_grid_mlp, X_train_std, y_train, "MLP Classifier")
    evaluate_and_log(results_log, "MLP Neural Net", model, X_test_std, y_test, cv_score, params)

    # 5. LDA
    try:
        model, cv_score, params = run_model_grid(LinearDiscriminantAnalysis(), param_grid_lda, X_train_std, y_train, "LDA")
        evaluate_and_log(results_log, "LDA", model, X_test_std, y_test, cv_score, params)
    except Exception as e:
        print(f"⚠️ Error en LDA (quizás colinealidad o pocas clases): {e}")

    # 6. Naive Bayes
    model, cv_score, params = run_model_grid(GaussianNB(), param_grid_nb, X_train_yj, y_train, "Naive Bayes")
    evaluate_and_log(results_log, "Naive Bayes", model, X_test_yj, y_test, cv_score, params)

    # --- G. Guardar Resultados ---
    print("\n>>> GUARDANDO RESULTADOS")
    df_results = pd.DataFrame(results_log)
    
    if not df_results.empty:
        # Ordenar por mejor Accuracy en Test
        df_results = df_results.sort_values(by="Test_Accuracy", ascending=False)
        
        output_file = "resultados_metricas_extinct_only.csv"
        df_results.to_csv(output_file, index=False)
        
        print(df_results)
        print(f"\n✅ Archivo generado: {output_file}")
    else:
        print("⚠️ No se generaron resultados (¿fallaron todos los modelos?).")