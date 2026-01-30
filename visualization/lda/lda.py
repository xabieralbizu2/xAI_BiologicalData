import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
INPUT_DIR = '.'  # Carpeta donde buscar los CSVs (usa '.' para la actual)
APPLY_CORR_FILTER = True
CORR_THRESHOLD = 0.9
APPLY_SCALING = True

# ---------------------------------------------------------
# 1. FUNCIONES AUXILIARES (TU CÓDIGO ORIGINAL)
# ---------------------------------------------------------
def get_uncorrelated_columns(df: pd.DataFrame, threshold: float = 0.95, visualization: bool = False) -> list:
    """Retorna la lista de columnas que NO tienen alta correlación entre sí."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [c for c in df.columns if c not in to_drop]

def draw_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Dibuja una elipse de covarianza (intervalo de confianza)."""
    if x.size < 2: return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)

def process_file(file_path):
    filename = os.path.basename(file_path)
    print(f"\n{'='*40}")
    print(f"PROCESANDO: {filename}")
    print(f"{'='*40}")

    try:
        # --- CARGA DE DATOS ---
        df = pd.read_csv(file_path)

        # 1. LIMPIEZA: Eliminar espacios en 'Status' si existe
        if 'Status' in df.columns:
            df['Status'] = df['Status'].astype(str).str.strip()
        else:
            print(f"⚠️ Saltando {filename}: No tiene columna 'Status'")
            return

        # 2. SEPARACIÓN DE CONJUNTOS
        df_extant = df[df['Status'] == 'Extant'].copy()
        df_extinct = df[df['Status'] == 'Extinct'].copy()

        # Si no hay Extant, no podemos entrenar
        if df_extant.empty:
            print(f"⚠️ Saltando {filename}: No hay datos 'Extant' para entrenar.")
            return

        # 3. SELECCIÓN DE VARIABLES NUMÉRICAS
        # Filtramos columnas que no sean numéricas automáticamente
        X_train = df_extant.select_dtypes(include=[np.number])
        y_train = df_extant['Target_Specie']

        X_project = df_extinct.select_dtypes(include=[np.number])
        y_project_specie = df_extinct['Target_Specie'] if not df_extinct.empty else pd.Series()

        # Asegurar que X_project tenga las mismas columnas que X_train
        # (Por si acaso alguna columna se borró en un subset y no en otro)
        common_cols = X_train.columns.intersection(X_project.columns)
        X_train = X_train[common_cols]
        X_project = X_project[common_cols]

        # ---------------------------------------------------------
        # PREPROCESAMIENTO
        # ---------------------------------------------------------
        features = list(X_train.columns)

        if APPLY_CORR_FILTER:
            cols_to_keep = get_uncorrelated_columns(X_train, threshold=CORR_THRESHOLD, visualization=False)
            X_train = X_train[cols_to_keep]
            X_project = X_project[cols_to_keep]
            features = cols_to_keep
            print(f"-> Variables tras filtro correlación: {len(features)}")

        if APPLY_SCALING:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            # Solo transformamos X_project si tiene datos
            if not X_project.empty:
                X_project_scaled = scaler.transform(X_project)
                X_project = pd.DataFrame(X_project_scaled, columns=features, index=X_project.index)
            
            X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)

        # ---------------------------------------------------------
        # ENTRENAMIENTO LDA
        # ---------------------------------------------------------
        n_classes = len(y_train.unique())
        n_features = X_train.shape[1]
        
        # Validar componentes
        n_components = min(2, n_classes - 1, n_features)
        if n_components < 2:
            print(f"⚠️ {filename}: No se pueden generar 2 dimensiones (Clases={n_classes}, Features={n_features}).")
            return

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda_train = lda.fit_transform(X_train, y_train)
        
        # Proyectar fósiles si existen
        X_lda_project = None
        if not X_project.empty:
            X_lda_project = lda.transform(X_project)

        # ---------------------------------------------------------
        # VISUALIZACIÓN (TU ESTILO ORIGINAL)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 9))

        # A. EXTANT (REFERENCIA) - CÍRCULOS + ELIPSES
        unique_extant = np.unique(y_train)
        cmap = plt.cm.get_cmap('tab10', len(unique_extant))

        for i, specie in enumerate(unique_extant):
            mask = (y_train == specie).values
            x_vals = X_lda_train[mask, 0]
            y_vals = X_lda_train[mask, 1]
            color = cmap(i)
            
            # Puntos
            ax.scatter(x_vals, y_vals, label=f"{specie}", color=color, alpha=0.6, s=30)
            
            # Elipse
            try:
                draw_ellipse(x_vals, y_vals, ax, n_std=2.0, edgecolor=color, facecolor=color, alpha=0.1)
            except:
                pass

        # B. EXTINCT (PROYECCIÓN) - ESTRELLAS
        if X_lda_project is not None:
            unique_project = np.unique(y_project_specie)
            markers_list = ['*', 'D', 'P', 'X', '^']
            
            for i, specie in enumerate(unique_project):
                mask = (y_project_specie == specie).values
                # Verificar si hay datos para esa especie
                if np.sum(mask) > 0:
                    x_vals = X_lda_project[mask, 0]
                    y_vals = X_lda_project[mask, 1]
                    marker = markers_list[i % len(markers_list)]
                    
                    ax.scatter(x_vals, y_vals, label=f"{specie} (Fossil)", 
                               marker=marker, s=120, c='yellow', edgecolors='black', linewidth=1, zorder=10)

        # C. DETALLES Y GUARDADO
        var_expl = lda.explained_variance_ratio_
        ax.set_xlabel(f"LD1 ({var_expl[0]:.2%} var)")
        ax.set_ylabel(f"LD2 ({var_expl[1]:.2%} var)")
        ax.set_title(f"LDA: {filename} (Extant Ref vs Project)")
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Guardar con el mismo nombre pero extensión .png
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(INPUT_DIR, output_filename)
        
        plt.savefig(output_path, dpi=300)
        plt.close() # Importante: cerrar para no mezclar con el siguiente
        print(f"✅ Gráfico guardado: {output_filename}")

    except Exception as e:
        print(f"❌ Error crítico procesando {filename}: {e}")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: No existe el directorio {INPUT_DIR}")
    else:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
        if not files:
            print("No se encontraron CSVs.")
        else:
            print(f"Encontrados {len(files)} archivos. Iniciando...")
            for f in files:
                process_file(os.path.join(INPUT_DIR, f))