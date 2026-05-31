import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv("src/dataset_nbool_std.csv")

feature_cols = df.select_dtypes(include="number").columns.tolist()
species_col  = "Target_Specie"

# ── Centroides por especie ────────────────────────────────────────────────────
centroids = df.groupby(species_col)[feature_cols].mean()
species   = centroids.index.tolist()
n         = len(species)

# ── Covarianza pooled within-class + pseudo-inversa ──────────────────────────
groups     = [df[df[species_col] == sp][feature_cols].values for sp in species]
pooled_cov = sum((g - g.mean(axis=0)).T @ (g - g.mean(axis=0)) for g in groups)
pooled_cov /= (len(df) - n)
pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6  # regularizacion ridge
VI = pinv(pooled_cov)

# ── Matriz de distancias de Mahalanobis ───────────────────────────────────────
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = mahalanobis(centroids.iloc[i], centroids.iloc[j], VI)
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

dist_df = pd.DataFrame(dist_matrix, index=species, columns=species)

print("Matriz de distancias:")
print(dist_df.round(4).to_string())

# ── Guardar la imagen de las distancias (Mapa de calor) ───────────────────────
# Configuramos el tamaño del gráfico (puedes aumentarlo si tienes muchas especies)
plt.figure(figsize=(10, 8))

# Creamos el mapa de calor
sns.heatmap(dist_df, 
            annot=True,          # Muestra los números en cada celda
            fmt=".2f",           # Formato a 2 decimales
            cmap="YlOrRd",       # Paleta de colores (Yellow-Orange-Red)
            cbar_kws={'label': 'Distancia de Mahalanobis'},
            linewidths=0.5, 
            linecolor='gray')

# Títulos y etiquetas
plt.title('Matriz de Distancias de Mahalanobis entre Especies', fontsize=14, pad=15)
plt.xlabel('Especie', fontsize=12)
plt.ylabel('Especie', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Inclinamos los nombres del eje X para que se lean bien
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar la imagen
nombre_imagen = "distancias_mahalanobis.png"
plt.savefig(nombre_imagen, dpi=300) # dpi=300 asegura que la imagen se guarde en alta resolución
plt.close() # Cerramos la figura para liberar memoria

print(f"\n¡Imagen guardada exitosamente como '{nombre_imagen}'!")