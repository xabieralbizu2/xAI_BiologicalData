import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE VARIABLES
# ---------------------------------------------------------
# Variables Numéricas (ISO + Furrow + SSFA)
ISO_VARS = [
    'Sq', 'Ssk', 'Sku', 'Sp', 'Sv', 'Sz', 'Sa', 'Sal (s = 0.2)', 'Std (Reference angle = 0°)', 
    'Sdq', 'Sdr', 'Smr (c = 1 µm below highest peak)', 'Smc (p = 10%)', 'Sdc (p = 10%, q = 90%)', 
    'Vm (p = 10%)', 'Vv (p = 10%)', 'Vmp (p = 10%)', 'Vmc (p = 10%, q = 80%)', 'Vvc (p = 10%, q = 80%)', 
    'Vvv (p = 80%)', 'Sk', 'Spk', 'Svk', 'Smrk1', 'Smrk2', 'Sak1', 'Sak2', 'Spkx', 'Svkx', 
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
FURROW_VARS = ['Maximum depth of furrows', 'Mean depth of furrows', 'Mean density of furrows']
SSFA_VARS = ['Fractal complexity (Asfc)', 'HAsfc81 (HAsfc81)', 'Median of Asfc (MedianAsfc)', 
             'Mean of Asfc (MeanAsfc)', 'Length-scale anisotropy (Sfrax) (epLsar)', 
             'Length-scale anisotropy (NewEplsar)']

ALL_NUMERIC_VARS = ISO_VARS + FURROW_VARS + SSFA_VARS

# Variables Categóricas (para One-Hot)
CAT_VARS = ['Teeth', 'Upper/Lower', 'Side', 'Sex']

# ---------------------------------------------------------
# 2. CARGA Y LIMPIEZA
# ---------------------------------------------------------
def load_and_clean_data(file_name):
    print(f"Cargando {file_name}...")
    
    # Detección de formato para evitar el error de Unicode
    if file_name.endswith('.xlsx'):
        df = pd.read_excel(file_name, header=1)
    else:
        df = pd.read_csv(file_name, header=1)
    
    # Limpieza de nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Limpieza de valores clave
    if 'Extant/Extinct' in df.columns:
        df['Extant/Extinct'] = df['Extant/Extinct'].astype(str).str.strip()
    
    # AQUÍ ESTABA EL FILTRO: LO HEMOS QUITADO
    # Antes: df_extinct = df[df['Extant/Extinct'] == 'Extinct'].copy()
    # Ahora: Usamos todo el dataframe
    print(f"Registros cargados (Total): {len(df)}")
    if 'Extant/Extinct' in df.columns:
        print(f" -> Extant: {len(df[df['Extant/Extinct']=='Extant'])}")
        print(f" -> Extinct: {len(df[df['Extant/Extinct']=='Extinct'])}")
    
    return df

# ---------------------------------------------------------
# 3. ONE-HOT ENCODING
# ---------------------------------------------------------
def apply_one_hot(df):
    print("\n--- Procesando Variables Categóricas ---")
    
    # 1. Limpieza específica de Sexo
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.lower().str.strip()
        df['Sex'] = df['Sex'].replace({'indet': 'unknown', 'nan': 'unknown'})
    
    for col in CAT_VARS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # print(f"Variable '{col}': {df[col].unique()}") # Descomentar para debug
    
    # 2. Separar Numéricas
    valid_numeric = [c for c in ALL_NUMERIC_VARS if c in df.columns]
    X_numeric = df[valid_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 3. Separar Categóricas
    valid_cat = [c for c in CAT_VARS if c in df.columns]
    X_cat = df[valid_cat].copy()
    
    # 4. Aplicar Get Dummies (One-Hot)
    # drop_first=True evita redundancia (multicolinealidad)
    X_cat_encoded = pd.get_dummies(X_cat, columns=valid_cat, drop_first=True, dtype=int)
    
    print(f"\nGeneradas {X_cat_encoded.shape[1]} nuevas columnas categóricas.")
    print("Columnas One-Hot:", X_cat_encoded.columns.tolist())
    
    # 5. Unir todo en X_final
    X_final = pd.concat([X_numeric, X_cat_encoded], axis=1)
    
    return X_final, df['Specie']

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Ajusta esta ruta si es necesario
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = os.path.join(BASE_DIR, "src/database_papionini_microtexture_3PAL.xlsx")
    
    # Lógica de búsqueda del archivo
    if not os.path.exists(FILE_NAME):
        # Intento buscar en el directorio actual si falla src/
        FILE_NAME = "database_papionini_microtexture_3PAL.xlsx"
    
    if os.path.exists(FILE_NAME):
        df_raw = load_and_clean_data(FILE_NAME)
        
        if df_raw.empty:
            print("Error: Dataset vacío.")
        else:
            X, y = apply_one_hot(df_raw)
            
            print("\n--- Guardando Resultado ---")
            
            # Crear el DataFrame final para exportar
            df_export = X.copy()
            
            # AÑADIMOS EL TARGET (Specie)
            df_export['Specie'] = y
            
            # AÑADIMOS LA COLUMNA STATUS (Extant/Extinct) 
            if 'Extant/Extinct' in df_raw.columns:
                df_export['Status'] = df_raw['Extant/Extinct']
            else:
                print("⚠️ Aviso: No se encontró la columna 'Extant/Extinct' original.")
            
            # Ordenamos columnas para que Status y Target queden al principio
            first_cols = ['Status', 'Target_Specie']
            existing_first = [c for c in first_cols if c in df_export.columns]
            rest_cols = [c for c in df_export.columns if c not in existing_first]
            
            df_export = df_export[existing_first + rest_cols]

            print(f"Dimensiones finales: {df_export.shape}")
            
            output_file = "dataset_complete_onehot.csv"
            df_export.to_csv(output_file, index=False)
            print(f"\n✅ Dataset COMPLETO guardado en: {output_file}")
            print(f"   (Incluye TODOS los registros: Extant + Extinct)")

    else:
        print(f"❌ Error crítico: No se encuentra el archivo {FILE_NAME}")