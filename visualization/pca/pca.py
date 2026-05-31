import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# ==========================================
# FUNCIÓN PARA DIBUJAR LA ELIPSE
# ==========================================
def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='black', **kwargs):
    """
    Dibuja una elipse de covarianza para un conjunto de puntos 2D (x, y).
    n_std=2.0 representa aproximadamente el 95% de confianza.
    """
    if x.size < 2 or y.size < 2:
        return # No se puede dibujar una elipse con 1 solo punto
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Calcular radios de la elipse basados en los autovalores
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    # Escalar y trasladar la elipse usando la media y desviación estándar
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ==========================================
# 1. CARGA DE DATOS Y FILTRADO (Solo Extant)
# ==========================================
ruta_archivo = 'src/dataset_con_grupos.csv' 
columna_objetivo = 'Target_Specie'

df = pd.read_csv(ruta_archivo)

# Filtrar solo "Extant" (limpiamos espacios en blanco por precaución)
col_extant = 'Extant/Extinct '
df = df[df[col_extant].str.strip() == 'Extant'].copy()

# Limpieza básica
df = df.dropna(subset=[columna_objetivo])
df = df.fillna(df.select_dtypes(include=[np.number]).mean())

# ==========================================
# 2. PREPROCESAMIENTO Y PCA
# ==========================================
y = df[columna_objetivo]
X = df.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)
varianza_explicada = pca.explained_variance_ratio_ * 100

df_pca = pd.DataFrame(data=X_pca, columns=['CP1', 'CP2'])
df_pca[columna_objetivo] = y.values

# ==========================================
# 3. VISUALIZACIÓN CON ELIPSES Y CENTROIDES
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Extraer los colores asignados por seaborn para que la elipse coincida con el punto
especies = df_pca[columna_objetivo].unique()
paleta = sns.color_palette('Set1', n_colors=len(especies))
color_dict = dict(zip(especies, paleta))

# 3.1 Dibujar los puntos base
sns.scatterplot(
    x='CP1', y='CP2', hue=columna_objetivo, data=df_pca,
    palette=color_dict, s=70, alpha=0.6, edgecolor='white', ax=ax
)

# 3.2 Iterar sobre cada especie para dibujar su elipse y centroide
for especie in especies:
    datos_especie = df_pca[df_pca[columna_objetivo] == especie]
    x_coords = datos_especie['CP1'].values
    y_coords = datos_especie['CP2'].values
    
    color_especie = color_dict[especie]
    
    # Dibujar la Elipse (n_std=2.0 significa 95% de confianza)
    draw_confidence_ellipse(x_coords, y_coords, ax, n_std=2.0, 
                            edgecolor=color_especie, facecolor=color_especie, 
                            alpha=0.15, linewidth=2)
    
    # Dibujar el Centroide (Media)
    centroide_x = np.mean(x_coords)
    centroide_y = np.mean(y_coords)
    ax.scatter(centroide_x, centroide_y, marker='X', color=color_especie, 
               s=250, edgecolor='black', linewidth=1.5, zorder=5)

# ==========================================
# 4. FORMATO FINAL DEL GRÁFICO
# ==========================================
ax.set_title('PCA por Especie (Extant) con Centroides y Elipses (95%)', fontsize=16, pad=20)
ax.set_xlabel(f'Componente Principal 1 ({varianza_explicada[0]:.1f}% varianza)', fontsize=12)
ax.set_ylabel(f'Componente Principal 2 ({varianza_explicada[1]:.1f}% varianza)', fontsize=12)

# Ajuste de la leyenda (evitamos duplicados de las elipses/centroides)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:len(especies)], labels=labels[:len(especies)], 
          title='Especie', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Guardar y mostrar
plt.savefig('pca_extant_elipses.png', dpi=300, bbox_inches='tight')
plt.show()