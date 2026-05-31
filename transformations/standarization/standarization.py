import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

def standardize_dataset(input_path, output_path):
    print(f"\n📂 Procesando archivo: {input_path}")
    
    if not Path(input_path).exists():
        print(f"❌ Error: No existe el archivo: {input_path}")
        return

    # 1. DETECCIÓN INTELIGENTE DEL ENCABEZADO (Busca la fila "ID")
    print("   Leyendo estructura...")
    try:
        # Leemos las primeras lineas para encontrar dónde empiezan los datos
        df_preview = pd.read_excel(input_path, header=None, nrows=20)
    except:
        # Fallback si es CSV
        df_preview = pd.read_csv(input_path, header=None, nrows=20)

    header_idx = 0
    found = False
    for idx, row in df_preview.iterrows():
        row_str = row.astype(str).str.cat(sep=" ")
        if "ID" in row_str and "Specie" in row_str:
            header_idx = idx
            found = True
            print(f"✅ Encabezado detectado en fila {idx + 1}")
            break
    
    # 2. CARGAR DATOS
    try:
        df = pd.read_excel(input_path, header=header_idx)
    except:
        df = pd.read_csv(input_path, header=header_idx)

    # Limpiar nombres de columnas (quitar espacios extra)
    df.columns = df.columns.str.strip()
    original_count = len(df)
    
    # 3. SEPARAR METADATOS Y NUMÉRICOS
    # Lista de columnas que sabemos que son texto/info (NO tocar)
    metadata_cols = [
        'ID', 'Extant/Extinct', 'Status', 'Specie', 'Teeth', 
        'Upper/Lower', 'Side', 'Sex', 'Localización', 'Unnamed: 0', 'Specimen'
    ]
    
    # Identificar columnas numéricas dinámicamente
    # Son las que NO están en metadata Y se pueden convertir a números
    cols_to_scale = []
    
    for col in df.columns:
        if any(meta_word in col for meta_word in metadata_cols):
            continue
            
        # Intentamos convertir a numérico para ver si es válido
        # errors='coerce' convierte texto a NaN
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # Si la columna tiene suficientes números (más del 50%), la escalamos
        if numeric_series.notna().sum() > (len(df) * 0.5):
            cols_to_scale.append(col)
            # Actualizamos la columna en el DF para asegurar que sea float limpio
            df[col] = numeric_series.fillna(0) # Rellenamos nulos con 0 para que StandardScaler no falle

    print(f"   Variables numéricas detectadas: {len(cols_to_scale)}")
    print(f"   Variables de metadatos (intactas): {len(df.columns) - len(cols_to_scale)}")

    # 4. APLICAR STANDARD SCALER
    if cols_to_scale:
        scaler = StandardScaler()
        # Transformamos y volvemos a asignar a las mismas columnas (mantenemos nombres)
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        print("⚠️ Advertencia: No se encontraron columnas numéricas.")

    # 5. GUARDAR COMO CSV (Igual que one_hot.py)
    # Asegurar que la carpeta existe
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Guardado exitosamente en CSV:\n   -> {output_path}")

    # Verificación rápida
    if cols_to_scale:
        print("\n--- Verificación (Medias ~0, Std ~1) ---")
        print(df[cols_to_scale[:3]].describe().loc[['mean', 'std']])


# --- EJECUCIÓN ---
if __name__ == "__main__":
    
    # Ubicación del script
    script_dir = Path(__file__).parent.resolve()
    
    # Rutas relativas robustas (Buscamos en ../src o ../../src)
    possible_src = [
        (script_dir / "../src").resolve(),
        (script_dir / "../../src").resolve()
    ]
    
    src_folder = None
    for p in possible_src:
        if p.exists():
            src_folder = p
            break
            
    if not src_folder:
        print("❌ Error: No encuentro la carpeta 'src'.")
        sys.exit()

    # Nombres de archivo
    input_file = src_folder / "dataset_onehot.csv" # Tu Excel original sucio
    
    # SI NO EXISTE EL EXCEL ORIGINAL, BUSCAMOS VARIANTES
    if not input_file.exists():
        # Intenta buscar 'PALAEO3_microtexture2.xlsx' (sin espacio)
        if (src_folder / "dataset_onehot.csv").exists():
            input_file = src_folder / "dataset_onehot.csv"
        else:
            print(f"❌ Error: No encuentro el archivo {input_file.name}")
            sys.exit()

    # SALIDA: Ahora es .csv
    output_file = src_folder / "dataset_onehot_std.csv"

    standardize_dataset(input_file, output_file)