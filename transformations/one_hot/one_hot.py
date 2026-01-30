import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------

# Variables que sabemos que SON categorías (para One-Hot)
# Nota: 'Localización' está aquí pero la filtraremos explícitamente en la función
CATEGORICAL_VARS = ['Teeth', 'Upper/Lower', 'Side', 'Sex', 'Localización']

# Variables que NO son características numéricas y debemos ignorar en la conversión
# (ID, Target, Status, etc.)
NON_FEATURE_COLS = ['ID', 'Specie', 'Extant/Extinct', 'Status']

TARGET_VAR = 'Specie'

def load_and_clean_data(file_path):
    """
    Carga el archivo saltando metadatos y limpiando columnas vacías.
    """
    try:
        # Saltamos las primeras filas de metadatos (ajusta skiprows si es necesario)
        df = pd.read_excel(file_path, skiprows=3, sheet_name="3ºOPTION")
        
        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how='all')
        
        # Limpiar espacios en nombres de columnas y eliminar saltos de línea
        df.columns = [str(c).strip().replace('\n', ' ') for c in df.columns]
        
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return pd.DataFrame()

def apply_one_hot_dynamic(df):
    """
    Detecta y procesa TODAS las variables numéricas automáticamente,
    y aplica One-Hot a las categorías seleccionadas.
    """
    print(f"Procesando {len(df)} registros...")
    
    # 1. Identificar columnas categóricas presentes (Excluyendo 'Localización' como pediste)
    # -------------------------------------------------------------------------
    cats_to_encode = [c for c in CATEGORICAL_VARS if c in df.columns and c != 'Localización']
    
    # 2. Identificar columnas numéricas (Lógica: Todo lo que NO sea cat, target o ID)
    # -------------------------------------------------------------------------
    # Lista de columnas a excluir del análisis numérico
    excluded_cols = set(cats_to_encode + NON_FEATURE_COLS + [TARGET_VAR])
    
    # Si 'Localización' existe en el df, añádemela a excluidas para que no intente convertirla a número
    if 'Localización' in df.columns:
        excluded_cols.add('Localización')

    # Las columnas restantes son las numéricas (ISO, SSFA, Pruning, Furrows, etc.)
    numeric_cols = [c for c in df.columns if c not in excluded_cols]
    
    print(f" -> Variables numéricas detectadas: {len(numeric_cols)}")
    print(f" -> Variables categóricas a codificar: {len(cats_to_encode)}")

    # 3. Convertir numéricas (Forzar a float, errores a NaN)
    # -------------------------------------------------------------------------
    for var in numeric_cols:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    # 4. Aplicar One-Hot Encoding
    # -------------------------------------------------------------------------
    X_cat = pd.get_dummies(df[cats_to_encode], prefix=cats_to_encode)
    
    # 5. Combinar todo
    # -------------------------------------------------------------------------
    X = pd.concat([df[numeric_cols], X_cat], axis=1)
    
    # 6. Extraer Target
    y = df[TARGET_VAR] if TARGET_VAR in df.columns else None
    
    return X, y

# ---------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    # Solución robusta para encontrar el archivo en la misma carpeta que el script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "PALAEO3_microtexture2.xlsx"
    file_path = os.path.join(base_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo en: {file_path}")
    else:
        print(f"Cargando datos desde: {file_path}")
        df_raw = load_and_clean_data(file_path)
        
        if df_raw.empty:
            print("Error: Dataset vacío o ilegible.")
        else:
            X, y = apply_one_hot_dynamic(df_raw)
            
            # Crear DataFrame final para exportar
            df_export = X.copy()
            
            if y is not None:
                df_export['Target_Specie'] = y
            
            # Preservar columna de Status si existe (Extant/Extinct)
            if 'Extant/Extinct' in df_raw.columns:
                df_export['Status'] = df_raw['Extant/Extinct']
            
            output_file = os.path.join(base_dir, "dataset_microtexture_full_vars.csv")
            df_export.to_csv(output_file, index=False)
            
            print(f"✅ Proceso completado.")
            print(f"✅ Dimensiones finales: {df_export.shape} (Antes eran ~46 columnas)")
            print(f"✅ Archivo guardado como: {output_file}")