import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------

# Variables que sabemos que SON categorías (para One-Hot)
CATEGORICAL_VARS = ['Teeth', 'Upper/Lower', 'Side', 'Sex', 'Localización']

# Variables que NO son características numéricas (Metadata)
# Se excluyen del One-Hot, pero ahora nos aseguraremos de guardarlas al final
NON_FEATURE_COLS = ['ID', 'Specie', 'Extant/Extinct', 'Status', 'Specimen']

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
    
    # 1. Identificar columnas categóricas presentes
    cats_to_encode = [c for c in CATEGORICAL_VARS if c in df.columns and c != 'Localización']
    
    # 2. Identificar columnas numéricas
    excluded_cols = set(cats_to_encode + NON_FEATURE_COLS + [TARGET_VAR])
    
    if 'Localización' in df.columns:
        excluded_cols.add('Localización')

    numeric_cols = [c for c in df.columns if c not in excluded_cols]
    
    print(f" -> Variables numéricas detectadas: {len(numeric_cols)}")
    print(f" -> Variables categóricas a codificar: {len(cats_to_encode)}")

    # 3. Convertir numéricas
    for var in numeric_cols:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    # 4. Aplicar One-Hot Encoding
    if cats_to_encode:
        X_cat = pd.get_dummies(df[cats_to_encode], prefix=cats_to_encode)
    else:
        X_cat = pd.DataFrame() # DataFrame vacío si no hay categóricas
    
    # 5. Combinar todo
    X = pd.concat([df[numeric_cols], X_cat], axis=1)
    
    # 6. Extraer Target
    y = df[TARGET_VAR] if TARGET_VAR in df.columns else None
    
    return X, y

# ---------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ubicación dinámica basada en dónde está el script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nombre del archivo Excel de entrada
    file_name = "PALAEO3_microtexture2.xlsx"
    file_path = os.path.join(base_dir, file_name)

    # Si no encuentra el archivo en la carpeta actual, intenta buscar en ../src/
    if not os.path.exists(file_path):
        alternate_path = os.path.join(base_dir, "../src", file_name)
        if os.path.exists(alternate_path):
            file_path = alternate_path

    if not os.path.exists(file_path):
        print(f"❌ Error: No se encuentra el archivo en: {file_path}")
    else:
        print(f"📂 Cargando datos desde: {file_path}")
        df_raw = load_and_clean_data(file_path)
        
        if df_raw.empty:
            print("❌ Error: Dataset vacío o ilegible.")
        else:
            X, y = apply_one_hot_dynamic(df_raw)
            
            # --- CREAR DATAFRAME FINAL ---
            df_export = X.copy()
            
            # 1. Reincorporar Target (Specie)
            if y is not None:
                df_export['Target_Specie'] = y
            
            # 2. Reincorporar Status (Extant/Extinct)
            if 'Extant/Extinct' in df_raw.columns:
                df_export['Status'] = df_raw['Extant/Extinct']
            
            # 3. REINCORPORAR ID / SPECIMEN (¡NUEVO!)
            # Buscamos columnas de identificación y las pegamos de vuelta
            id_cols = ['ID', 'Specimen', 'Label', 'Muestra']
            
            found_id = False
            for col in id_cols:
                if col in df_raw.columns:
                    # Copiamos la columna del original al exportable
                    df_export[col] = df_raw[col]
                    print(f"   -> Columna preservada: {col}")
                    found_id = True
            
            # Si 'ID' existe pero los modelos piden 'Specimen', hacemos una copia extra llamada 'Specimen'
            # para evitar errores en scripts futuros
            if 'ID' in df_export.columns and 'Specimen' not in df_export.columns:
                df_export['Specimen'] = df_export['ID']
                print("   -> (Se creó copia de 'ID' como 'Specimen' para compatibilidad)")

            # Guardar
            output_file = os.path.join(os.path.dirname(file_path), "dataset_onehot.csv")
            df_export.to_csv(output_file, index=False)
            
            print(f"\n✅ Proceso completado.")
            print(f"✅ Dimensiones finales: {df_export.shape}")
            print(f"✅ Archivo guardado como: {output_file}")