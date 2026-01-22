import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# --- CONFIGURACIÓN ---
APPLY_CORR_FILTER = True  # <--- Flag: Cambia a False para desactivar el filtrado
CORR_THRESHOLD = 0.95     # Umbral de correlación

# --- FUNCIÓN DE FILTRADO ---
def transform_under_corr(df: pd.DataFrame, threshold: float = 0.95, visualization: bool = False) -> pd.DataFrame:
    """
    Elimina columnas que tienen una correlación absoluta mayor al umbral especificado.
    """
    # Calculamos la matriz de correlación (valores absolutos)
    corr_matrix = df.corr().abs()
    
    # Seleccionamos el triángulo superior de la matriz de correlación
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identificamos columnas con correlación mayor al umbral
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if visualization:
        print(f"--- FILTRO DE CORRELACIÓN ---")
        print(f"Umbral: {threshold}")
        print(f"Variables originales: {df.shape[1]}")
        print(f"Variables a eliminar ({len(to_drop)}): {to_drop}")
        
    return df.drop(columns=to_drop)

# 1. Cargar datos y filtrar solo los "Extinct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asegúrate de que la ruta sea correcta según tu estructura de carpetas
FILE = os.path.join(BASE_DIR, "../src/dataset_complete_onehot.csv")

df = pd.read_csv(FILE)
df_extinct = df[df['Status'] == 'Extinct'].copy()

# 2. Separar características (X) y objetivo (y)
X = df_extinct.drop(columns=['Status', 'Specie'])
y = df_extinct['Specie']

# --- APLICACIÓN DEL FILTRO (NUEVO) ---
if APPLY_CORR_FILTER:
    # Se aplica solo a X, ya que no queremos correlacionar con el target categórico aquí
    X = transform_under_corr(X, threshold=CORR_THRESHOLD, visualization=True)
    if APPLY_CORR_FILTER and hasattr(X, "shape"):
        print(f"Variables finales: {X.shape[1]}")
        print("-----------------------------")

# 3. Entrenar LDA
# Nota: Si tras el filtrado quedan menos features que n_components, LDA ajustará automáticamente,
# pero es buena práctica no pedir más componentes de los posibles (min(n_classes - 1, n_features))
n_classes = len(y.unique())
n_features = X.shape[1]
n_components = min(2, n_classes - 1, n_features)

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# 4. Crear DataFrame con las proyecciones
# Ajustamos nombres de columnas dinámicamente por si n_components cambia
col_names = [f"LD{i+1}" for i in range(X_lda.shape[1])]
lda_df = pd.DataFrame(X_lda, columns=col_names)
lda_df["target"] = y.values

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
fig, ax = plt.subplots(figsize=(10, 8))
targets = lda_df["target"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))

# Verificamos si tenemos al menos 2 dimensiones para graficar en 2D
if X_lda.shape[1] >= 2:
    for i, target in enumerate(targets):
        subset = lda_df[lda_df["target"] == target]
        ax.scatter(subset["LD1"], subset["LD2"], label=target, color=colors[i], alpha=0.7)
        try:
            draw_ellipse(subset["LD1"], subset["LD2"], ax, n_std=2.0, 
                         edgecolor=colors[i], facecolor=colors[i], alpha=0.2)
        except:
            pass
    
    var_1 = lda.explained_variance_ratio_[0] if len(lda.explained_variance_ratio_) > 0 else 0
    var_2 = lda.explained_variance_ratio_[1] if len(lda.explained_variance_ratio_) > 1 else 0
    
    ax.set_xlabel(f"LD1 ({var_1:.2%} varianza)")
    ax.set_ylabel(f"LD2 ({var_2:.2%} varianza)")
else:
    print("Advertencia: LDA resultó en menos de 2 dimensiones. No se puede generar scatterplot 2D estándar.")

title_suffix = " (Con reducción de correlación)" if APPLY_CORR_FILTER else " (Todas las variables)"
ax.set_title(f"LDA - Especies Extintas{title_suffix}")
ax.legend()
plt.tight_layout()

plt.savefig("lda_norm.png")
print("Imagen guardada como 'lda_norm.png'")

plt.show()