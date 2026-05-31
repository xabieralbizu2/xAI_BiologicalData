import pandas as pd
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer
import os

def perform_dagostino_test(df, numeric_cols):
    """
    Ejecuta el test de D'Agostino-Pearson sobre las columnas dadas.
    Retorna un DataFrame con los estadísticos y p-values, ORDENADO por p-value.
    """
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        
        # El test requiere al menos 8 muestras
        if len(data) >= 8:
            stat, p_value = normaltest(data)
            
            # Interpretación rápida (p < 0.05 -> No es normal)
            is_normal = p_value >= 0.05
            
            results.append({
                'Variable': col,
                'Statistic': stat,
                'p-value': p_value,
                'Is_Normal (p>0.05)': is_normal
            })
        else:
            results.append({
                'Variable': col,
                'Statistic': np.nan,
                'p-value': np.nan,
                'Is_Normal (p>0.05)': 'Not enough data'
            })
            
    df_res = pd.DataFrame(results)
    
    # ORDENAR: De menor a mayor p-value.
    # Los primeros serán los MENOS normales (p muy bajo).
    # Los últimos serán los MÁS normales (p alto).
    return df_res.sort_values(by='p-value', ascending=True)

if __name__ == "__main__":
    # 1. CARGA DE DATOS
    # -----------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE = os.path.join(BASE_DIR, "dataset_onehot.csv")

    df = pd.read_csv(FILE)

    # 2. SELECCIÓN DE COLUMNAS CONTINUAS
    # ----------------------------------
    # Excluimos columnas categóricas, texto, y las one-hot (binarias)
    cols_to_exclude = ['Status', 'Specie', 'Target_Specie']
    
    # Detectamos las binarias (One-Hot) automáticamente: valores solo 0 y 1
    # Esto es importante porque Yeo-Johnson no tiene sentido en binarias
    binary_cols = [c for c in df.select_dtypes(include=np.number).columns 
                   if df[c].nunique() <= 2 and set(df[c].unique()).issubset({0, 1})]
    
    continuous_cols = [c for c in df.select_dtypes(include=np.number).columns 
                       if c not in cols_to_exclude and c not in binary_cols]
    
    print(f"Variables continuas detectadas para normalizar: {len(continuous_cols)}")

    # 3. TEST DE NORMALIDAD (ANTES)
    # -----------------------------
    print("Ejecutando Test D'Agostino Pearson (Pre-Transformación)...")
    df_test_before = perform_dagostino_test(df, continuous_cols)
    
    output_test_before = "normality_test_before.csv"
    df_test_before.to_csv(output_test_before, index=False)
    print(f" -> Guardado (Ordenado por p-value): {output_test_before}")

    # 4. TRANSFORMACIÓN (YEO-JOHNSON)
    # -------------------------------
    print("Aplicando Transformación Yeo-Johnson...")
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    
    # Transformamos solo las continuas
    X_trans = pt.fit_transform(df[continuous_cols])
    
    # Creamos el nuevo dataframe normalizado
    df_normalized = df.copy()
    df_normalized[continuous_cols] = X_trans
    
    output_normalized = "dataset_normalized.csv"
    df_normalized.to_csv(output_normalized, index=False)
    print(f" -> Guardado: {output_normalized}")

    # 5. TEST DE NORMALIDAD (DESPUÉS)
    # -------------------------------
    print("Ejecutando Test D'Agostino Pearson (Post-Transformación)...")
    df_test_after = perform_dagostino_test(df_normalized, continuous_cols)
    
    output_test_after = "normality_test_after.csv"
    df_test_after.to_csv(output_test_after, index=False)
    print(f" -> Guardado (Ordenado por p-value): {output_test_after}")

    # Resumen rápido
    normal_before = df_test_before['Is_Normal (p>0.05)'].sum()
    normal_after = df_test_after['Is_Normal (p>0.05)'].sum()
    
    print(f"\n✅ Proceso completado.")
    print(f"Variables normales antes: {normal_before}")
    print(f"Variables normales después: {normal_after}")