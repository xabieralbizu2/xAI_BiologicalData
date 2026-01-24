import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# --- CONFIGURACIÓN ---
APPLY_CORR_FILTER = False  
CORR_THRESHOLD = 0.90     

# --- FUNCIÓN DE FILTRADO ---
def get_uncorrelated_columns(df: pd.DataFrame, threshold: float = 0.95, visualization: bool = False) -> list:
    """
    Identifica las columnas a mantener basándose en la correlación.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if visualization:
        print(f"--- FILTRO DE CORRELACIÓN (Sobre datos de entrenamiento) ---")
        print(f"Umbral: {threshold}")
        print(f"Variables originales: {df.shape[1]}")
        print(f"Variables a eliminar ({len(to_drop)}): {to_drop}")
        
    return [c for c in df.columns if c not in to_drop]

# 1. Cargar datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "../src/dataset_complete_onehot_norm.csv")

df = pd.read_csv(FILE)

# Limpieza básica (asegurar numéricos)
df.columns = df.columns.str.strip()
cols_meta = ['Status', 'Specie', 'Specimen'] # Columnas que no son features
features = [c for c in df.columns if c not in cols_meta]

# Convertir a numérico y rellenar nulos
df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. Separar EXTANT (Entrenamiento) y EXTINCT (Proyección)
df_extant = df[df['Status'] == 'Extant'].copy()
df_extinct = df[df['Status'] == 'Extinct'].copy()

X_train = df_extant[features]
y_train = df_extant['Specie']

X_project = df_extinct[features]
y_project_specie = df_extinct['Specie']

print(f"Datos Extant (Train): {X_train.shape}")
print(f"Datos Extinct (Project): {X_project.shape}")

# --- APLICACIÓN DEL FILTRO DE CORRELACIÓN ---
if APPLY_CORR_FILTER:
    # Aprendemos qué columnas borrar basándonos SOLO en Extant
    cols_to_keep = get_uncorrelated_columns(X_train, threshold=CORR_THRESHOLD, visualization=True)
    
    # Aplicamos el filtro a ambos
    X_train = X_train[cols_to_keep]
    X_project = X_project[cols_to_keep]
    
    print(f"Variables finales utilizadas: {X_train.shape[1]}")
    print("-----------------------------")

# 3. Entrenar LDA (Con Extant)
n_classes = len(y_train.unique())
n_features = X_train.shape[1]
n_components = min(2, n_classes - 1, n_features)

lda = LinearDiscriminantAnalysis(n_components=n_components)

# Ajustamos (Fit) con Extant
X_lda_train = lda.fit_transform(X_train, y_train)

# Proyectamos (Transform) los Extinct usando el modelo entrenado
# Nota: Si Extinct tiene valores muy locos, saldrán muy lejos en el gráfico
X_lda_project = lda.transform(X_project)

# 4. Crear DataFrames para graficar
# Extant
col_names = [f"LD{i+1}" for i in range(X_lda_train.shape[1])]
df_plot_train = pd.DataFrame(X_lda_train, columns=col_names)
df_plot_train["target"] = y_train.values

# Extinct
df_plot_project = pd.DataFrame(X_lda_project, columns=col_names)
df_plot_project["target"] = y_project_specie.values # Para etiquetarlos si quieres

# 5. Función para graficar elipses
def draw_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if x.size < 2: return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)

# 6. Generar el gráfico
fig, ax = plt.subplots(figsize=(12, 10))

# Colores para las especies Extant
targets_train = df_plot_train["target"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(targets_train)))

if X_lda_train.shape[1] >= 2:
    # A. Graficar EXTANT (Puntos + Elipses)
    for i, target in enumerate(targets_train):
        subset = df_plot_train[df_plot_train["target"] == target]
        
        # Puntos Extant
        ax.scatter(subset["LD1"], subset["LD2"], label=f"{target} (Extant)", 
                   color=colors[i], alpha=0.6, s=30)
        
        # Elipse de confianza (Standard Deviation)
        try:
            draw_ellipse(subset["LD1"], subset["LD2"], ax, n_std=2.0, 
                         edgecolor=colors[i], facecolor=colors[i], alpha=0.1)
        except:
            pass

    # B. Graficar EXTINCT (Proyección)
    # Los pintamos en negro (o gris oscuro) con un marcador diferente ('x' o '^')
    # para ver dónde caen respecto a las vivas.
    ax.scatter(df_plot_project["LD1"], df_plot_project["LD2"], 
               c='black', marker='x', s=50, label='Extinct (Proyectado)', alpha=0.9)
    
    # Opcional: Etiquetar los puntos Extinct con su nombre de especie si no son demasiados
    # for idx, row in df_plot_project.iterrows():
    #     ax.text(row['LD1'], row['LD2'], row['target'], fontsize=8, alpha=0.7)

    # Info de varianza explicada
    var_1 = lda.explained_variance_ratio_[0] if len(lda.explained_variance_ratio_) > 0 else 0
    var_2 = lda.explained_variance_ratio_[1] if len(lda.explained_variance_ratio_) > 1 else 0
    
    ax.set_xlabel(f"LD1 ({var_1:.2%} varianza explicada)")
    ax.set_ylabel(f"LD2 ({var_2:.2%} varianza explicada)")

else:
    print("Advertencia: LDA resultó en menos de 2 dimensiones.")

title_suffix = " (Entrenado en Extant -> Proyectado Extinct)"
ax.set_title(f"Espacio LDA{title_suffix}")
ax.legend(loc='best', fontsize='small')
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig("lda_extant_projected.png")
print("Imagen guardada como 'lda_extant_projected.png'")

plt.show()