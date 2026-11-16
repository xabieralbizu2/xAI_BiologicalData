import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from scipy.stats import entropy
import seaborn as sns

# ================= CONFIGURACIÓN =================
# Cambia esto por la ruta de tu archivo
ARCHIVO_EXCEL = os.path.join('src', 'PALAEO3_microtexture2.xlsx')
ARCHIVO_CSV = os.path.join('src', 'dataset_normalized.csv')
HOJA = 'ALL'
HEADER = 3
COLUMNA_ESTADO = 'Extant/Extinct ' 

# Nuevos nombres de salida
SALIDA_EMD = 'distancias_emd.csv'
SALIDA_PVALOR = 'significancia_pvalor.csv'
SALIDA_GMM = 'verosimilitud_gmm'
# =================================================

def calcular_emd(data1, data2):
    """Calcula la Earth Mover's Distance con datos estandarizados."""
    try:
        d1, d2 = np.array(data1), np.array(data2)
        if len(d1) < 2 or len(d2) < 2: return np.nan
        combinado = np.concatenate([d1, d2])
        mu, sigma = np.mean(combinado), np.std(combinado)
        if sigma == 0: return 0
        return stats.wasserstein_distance((d1 - mu) / sigma, (d2 - mu) / sigma)
    except:
        return np.nan
    
def generar_analisis_gmm(df, grupo_base, grupo_evaluar, nombre_base, nombre_evaluar, columnas_num, index=0):
    """
    Entrena un GMM con 'grupo_base' y evalúa tanto al base como al 'grupo_evaluar'.
    Incluye el cálculo de ELPD (Leave-One-Out y Out-of-Sample) para una evaluación 
    predictiva rigurosa, evitando el sobreajuste.
    """
    print(f"\nIniciando análisis GMM multivariante (Modelo entrenado con: {nombre_base})...")
    
    # 1. Preparar datos
    X_base = grupo_base[columnas_num].dropna()
    X_evaluar = grupo_evaluar[columnas_num].dropna()
    
    scaler = StandardScaler()
    X_base_std = scaler.fit_transform(X_base)
    X_evaluar_std = scaler.transform(X_evaluar)
    
    # 2. Configurar y Entrenar GMM Clásico
    # Corregimos el bug del código original: ajustamos n_components dinámicamente 
    # para evitar errores si el grupo a analizar es pequeño.
    n_components = min(7, max(1, len(X_base_std) // 3))
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_base_std)
    
    # 3. Calcular Log-Likelihood punto por punto (Verosimilitud tradicional para el CSV)
    ll_base = gmm.score_samples(X_base_std)
    ll_evaluar = gmm.score_samples(X_evaluar_std)
    
    # === NUEVO: CÁLCULO DE ELPD ===
    
    # A) ELPD Cruzado (Out-of-Sample): Capacidad de predecir al OTRO grupo
    elpd_cruzado = np.mean(ll_evaluar)
    
    # B) ELPD LOO-CV (Leave-One-Out): Capacidad real de predecir su PROPIO grupo sin trampa
    loo = LeaveOneOut()
    log_preds_loo = []
    
    for train_index, test_index in loo.split(X_base_std):
        X_train, X_test = X_base_std[train_index], X_base_std[test_index]
        gmm_loo = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm_loo.fit(X_train)
        # Predecimos la probabilidad del ÚNICO punto que dejamos fuera
        log_prob = gmm_loo.score_samples(X_test)[0] 
        log_preds_loo.append(log_prob)
        
    elpd_loo = np.mean(log_preds_loo)
    
    # 4. Imprimir Resultados en Consola
    print(f"   ► Componentes (subgrupos) usados: {n_components}")
    print(f"   ► ELPD LOO-CV (Coherencia interna real de {nombre_base}): {elpd_loo:.4f}")
    print(f"   ► ELPD Cruzado ({nombre_base} prediciendo a {nombre_evaluar}): {elpd_cruzado:.4f}")
    
    # Diferencia entre predecirse a sí mismo y predecir al otro grupo
    diff_elpd = elpd_loo - elpd_cruzado
    print(f"   ► Diferencia predictiva (Penalización por cambiar de grupo): {diff_elpd:.4f}")
    
    # 5. Crear dataframe y guardar (Mantenemos tu salida original intacta)
    res_base = pd.DataFrame({'Estado': nombre_base, 'Log_Likelihood': ll_base})
    res_evaluar = pd.DataFrame({'Estado': nombre_evaluar, 'Log_Likelihood': ll_evaluar})
    df_gmm = pd.concat([res_base, res_evaluar])
    
    # Calcular EMD global
    emd_gmm = stats.wasserstein_distance(ll_base, ll_evaluar)
    
    # Guardar resultados
    df_gmm.to_csv(f"{SALIDA_GMM}_{index}.csv", index=False)
    
    print(f"✅ Análisis GMM guardado en: {SALIDA_GMM}_{index}.csv")
    print(f"   Distancia EMD de verosimilitud global: {emd_gmm:.4f}\n")

def generar_reportes_modularizados():
    try:
        # df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA, engine='openpyxl', header=HEADER)
        df = pd.read_csv(ARCHIVO_CSV)
        df[COLUMNA_ESTADO] = df[COLUMNA_ESTADO].astype(str).str.strip()

        grupo_extinct = df[df[COLUMNA_ESTADO].str.lower() == 'extinct']
        grupo_extant = df[df[COLUMNA_ESTADO].str.lower() == 'extant']

        resultados_completos = []
        columnas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        for col in columnas_num:
            d_fossil = grupo_extinct[col].dropna().values
            d_vivos = grupo_extant[col].dropna().values
            if len(d_fossil) < 3 or len(d_vivos) < 3: continue

            # Test de normalidad y elección de prueba
            _, p_norm1 = stats.shapiro(d_fossil)
            _, p_norm2 = stats.shapiro(d_vivos)
            if p_norm1 > 0.05 and p_norm2 > 0.05:
                _, p_levene = stats.levene(d_fossil, d_vivos)
                _, p_val = stats.ttest_ind(d_fossil, d_vivos, equal_var=(p_levene > 0.05))
            else:
                _, p_val = stats.mannwhitneyu(d_fossil, d_vivos, alternative='two-sided')

            emd = calcular_emd(d_fossil, d_vivos)
            
            resultados_completos.append({
                'Variable': col,
                'Distancia_EMD': round(emd, 4),
                'P_Value': round(p_val, 6),
                'Significativo': 'SÍ' if p_val < 0.05 else 'NO'
            })

        # --- MODULARIZACIÓN ---
        df_base = pd.DataFrame(resultados_completos)

        # 1. CSV de Distancias (Ordenado de más diferente a más igual)
        df_emd = df_base[['Variable', 'Distancia_EMD']].sort_values(by='Distancia_EMD', ascending=False)
        df_emd.to_csv(SALIDA_EMD, index=False)

        # 2. CSV de Significancia (Ordenado de más significativo a menos)
        df_p = df_base[['Variable', 'P_Value', 'Significativo']].sort_values(by='P_Value')
        df_p.to_csv(SALIDA_PVALOR, index=False)

        print(f"✅ Archivos generados con éxito:")
        print(f"   - {SALIDA_EMD} (Ranking por magnitud de cambio)")
        print(f"   - {SALIDA_PVALOR} (Ranking por evidencia estadística)")

        # --- ANÁLISIS GMM ---
        generar_analisis_gmm(df, grupo_extant, grupo_extinct, 'Extant', 'Extinct', columnas_num, index=1)
        generar_analisis_gmm(df, grupo_extinct, grupo_extant, 'Extinct', 'Extant', columnas_num, index=2)

        # --- ANÁLISIS GMM ---
        comparar_estructuras_pca(grupo_extant, grupo_extinct, columnas_num)


    except Exception as e:
        print(f"❌ Error: {e}")


def comparar_estructuras_pca(grupo_extant, grupo_extinct, columnas_num):
    """
    Compara la estructura de Extant y Extinct calculando la Importancia Global 
    de cada variable, combinando PC1 y PC2 para evitar problemas de signos 
    o desplazamientos entre componentes.
    """
    print("\nIniciando Análisis de Estructura Interna con PCA (Extant vs Extinct)...")
    
    X_extant = grupo_extant[columnas_num].dropna()
    X_extinct = grupo_extinct[columnas_num].dropna()
    
    scaler_extant = StandardScaler()
    X_extant_std = scaler_extant.fit_transform(X_extant)
    
    scaler_extinct = StandardScaler()
    X_extinct_std = scaler_extinct.fit_transform(X_extinct)
    
    # Entrenar los modelos PCA
    pca_extant = PCA(n_components=7, random_state=42)
    pca_extant.fit(X_extant_std)
    
    pca_extinct = PCA(n_components=7, random_state=42)
    pca_extinct.fit(X_extinct_std)
    
    # === LA MAGIA MATEMÁTICA CORREGIDA ===
    # 1. Extraer varianza explicada (cuánto pesa cada PC)
    var_extant = pca_extant.explained_variance_ratio_
    var_extinct = pca_extinct.explained_variance_ratio_
    
    # 2. Calcular la importancia de cada variable elevando al cuadrado (mata el signo negativo)
    # y multiplicando por la varianza del componente (combina PC1 y PC2)
    var_extant = pca_extant.explained_variance_ratio_
    var_extinct = pca_extinct.explained_variance_ratio_
    
    # 2. Calcular la importancia de cada variable acumulando los 7 componentes
    importancia_extant = np.zeros(X_extant_std.shape[1])
    importancia_extinct = np.zeros(X_extinct_std.shape[1])
    
    for i in range(7):  # Suma desde el PC1 (índice 0) hasta el PC7 (índice 6)
        importancia_extant += (pca_extant.components_[i]**2 * var_extant[i])
        importancia_extinct += (pca_extinct.components_[i]**2 * var_extinct[i])    
    # 3. Normalizar para que las importancias sumen 1 (100%) en ambos grupos
    importancia_extant = importancia_extant / importancia_extant.sum()
    importancia_extinct = importancia_extinct / importancia_extinct.sum()
    
    # Crear DataFrame
    df_importancia = pd.DataFrame({
        'Variable': columnas_num,
        'Importancia_Global_Extant': importancia_extant,
        'Importancia_Global_Extinct': importancia_extinct
    })
    
    # Calcular la diferencia absoluta real de su importancia en el sistema
    df_importancia['Diferencia_Estructural'] = abs(df_importancia['Importancia_Global_Extant'] - df_importancia['Importancia_Global_Extinct'])
    df_importancia['Porcentaje_Extant (%)'] = df_importancia['Importancia_Global_Extant'] * 100
    df_importancia['Porcentaje_Extinct (%)'] = df_importancia['Importancia_Global_Extinct'] * 100

    # Calcular cuántas veces mayor es el mayor respecto al menor (Fold Change)
    df_importancia['Veces_de_Diferencia (Ratio)'] = df_importancia[['Importancia_Global_Extant', 'Importancia_Global_Extinct']].max(axis=1) / df_importancia[['Importancia_Global_Extant', 'Importancia_Global_Extinct']].min(axis=1)
    
    # Ordenar de mayor a menor
    df_importancia = df_importancia.sort_values(by='Diferencia_Estructural', ascending=False)
    
    salida_pca = 'importancia_variables_pca.csv'
    df_importancia.to_csv(salida_pca, index=False)
    print(f"✅ Análisis de Importancia Global PCA guardado en: {salida_pca}")
    print("   (Contempla PC1 y PC2 ponderados sin problemas de signos)")

def plotear_varianza_acumulada_grupos():
    """
    Calcula y grafica el Porcentaje de Varianza Acumulada
    para los componentes principales del 2 al 7, comparando
    los grupos Extant y Extinct por separado.
    """
    print("\nGenerando gráfico de varianza acumulada (PC2 a PC7) para Extant vs Extinct...")

    df = pd.read_csv(ARCHIVO_CSV)
    df[COLUMNA_ESTADO] = df[COLUMNA_ESTADO].astype(str).str.strip()

    grupo_extinct = df[df[COLUMNA_ESTADO].str.lower() == 'extinct']
    grupo_extant = df[df[COLUMNA_ESTADO].str.lower() == 'extant']

    columnas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # 1. Preparar datos Extant
    X_extant = grupo_extant[columnas_num].dropna()
    scaler_extant = StandardScaler()
    X_extant_std = scaler_extant.fit_transform(X_extant)
    
    # 2. Preparar datos Extinct
    X_extinct = grupo_extinct[columnas_num].dropna()
    scaler_extinct = StandardScaler()
    X_extinct_std = scaler_extinct.fit_transform(X_extinct)
    
    # Asegurar que entrenamos al menos 7 componentes (si los datos lo permiten)
    n_comps = min(7, len(columnas_num), X_extant_std.shape[0], X_extinct_std.shape[0])
    
    # 3. Entrenar los modelos PCA
    pca_extant = PCA(n_components=n_comps, random_state=42)
    pca_extant.fit(X_extant_std)
    
    pca_extinct = PCA(n_components=n_comps, random_state=42)
    pca_extinct.fit(X_extinct_std)
    
    # 4. Calcular varianza acumulada (%)
    var_acum_extant = np.cumsum(pca_extant.explained_variance_ratio_) * 100
    var_acum_extinct = np.cumsum(pca_extinct.explained_variance_ratio_) * 100
    
    if n_comps < 2:
        print("No hay suficientes componentes/datos para graficar desde PC2.")
        return

    # 5. Preparar datos para el gráfico (del PC2 al PC7 -> índices 1 a n_comps-1)
    componentes_x = np.arange(2, n_comps + 1)
    y_extant = var_acum_extant[1:n_comps]
    y_extinct = var_acum_extinct[1:n_comps]
    
    # 6. Crear el gráfico comparativo
    plt.figure(figsize=(10, 6))
    
    # Línea Extant
    plt.plot(componentes_x, y_extant, marker='o', linestyle='-', color='teal', 
             linewidth=2.5, markersize=8, label='Extant')
    # Línea Extinct
    plt.plot(componentes_x, y_extinct, marker='s', linestyle='-', color='coral', 
             linewidth=2.5, markersize=8, label='Extinct')
    
    # Añadir las etiquetas numéricas a cada punto para mayor claridad
    for x, y_ext, y_exti in zip(componentes_x, y_extant, y_extinct):
        plt.text(x, y_ext + 1.2, f'{y_ext:.1f}%', ha='center', va='bottom', color='teal', fontweight='bold')
        plt.text(x, y_exti - 2.5, f'{y_exti:.1f}%', ha='center', va='top', color='coral', fontweight='bold')
        
    # Detalles estéticos del gráfico
    plt.title('Porcentaje de Varianza Acumulada (PC2 a PC7) - Extant vs Extinct', fontsize=14, pad=15)
    plt.xlabel('Número de Componentes Principales Acumulados', fontsize=12)
    plt.ylabel('Varianza Explicada Acumulada (%)', fontsize=12)
    plt.xticks(componentes_x)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajustar márgenes dinámicamente
    min_y = min(min(y_extant), min(y_extinct))
    max_y = max(max(y_extant), max(y_extinct))
    plt.ylim(min_y - 6, min(105, max_y + 6))
    
    plt.tight_layout()
    
    # Guardar archivo
    nombre_archivo = 'varianza_acumulada_pca_grupos.png'
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico comparativo guardado exitosamente como: {nombre_archivo}")


def calcular_entropia_combinaciones():
    """
    Calcula la entropía de Shannon para las distribuciones de combinaciones 
    one-hot (Teeth, Upper/Lower, Side, Sex) por cada Especie.
    """
    print("\nCalculando Entropía de las distribuciones one-hot por Especie...")

    df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA, header=HEADER)
    df[COLUMNA_ESTADO] = df[COLUMNA_ESTADO].astype(str).str.strip()

    
    # Columnas categóricas que definen el "perfil" del espécimen
    cat_cols = ['Teeth', 'Upper/Lower', 'Side', 'Sex']
    
    # Asegurarnos de que las columnas existen y quitar NaNs
    columnas_presentes = [c for c in cat_cols if c in df.columns]
    df_subset = df[['Specie'] + columnas_presentes].dropna()
    
    # Convertir a variables one-hot
    df_onehot = pd.get_dummies(df_subset, columns=columnas_presentes)
    df_onehot['Specie'] = df_subset['Specie']
    
    # Agrupar las columnas one-hot en una tupla para crear una "combinación única"
    onehot_cols = [c for c in df_onehot.columns if c != 'Specie']
    df_onehot['Combinacion'] = df_onehot[onehot_cols].apply(lambda row: tuple(row), axis=1)
    
    # Calcular frecuencias y entropía por especie
    resultados = []
    
    for especie, grupo in df_onehot.groupby('Specie'):
        conteos = grupo['Combinacion'].value_counts()
        probabilidades = conteos / conteos.sum()
        
        # Entropía en base 2 (Bits)
        entropia_valor = entropy(probabilidades, base=2)
        
        resultados.append({
            'Especie': especie,
            'N_Individuos': len(grupo),
            'N_Combinaciones_Unicas': len(conteos),
            'Entropia (Bits)': entropia_valor
        })
        
    df_resultados = pd.DataFrame(resultados).sort_values('Entropia (Bits)', ascending=False)
    
    # Guardar a CSV
    nombre_archivo = 'entropia_combinaciones_especies.csv'
    df_resultados.to_csv(nombre_archivo, index=False)
    print(f"✅ Análisis de Entropía guardado exitosamente en: {nombre_archivo}")
    
    return df_resultados


def plot_distribuciones_por_especie(variables, col_especie='Specie'):
    """
    Genera gráficos de distribución para una lista de variables separadas por especie.
    Detecta automáticamente si la variable es booleana/categórica o continua.
    
    Parámetros:
    - df: DataFrame de Pandas con los datos.
    - variables: Lista de strings con los nombres de las columnas a plotear.
    - col_especie: Nombre de la columna que contiene las especies.
    """
    df = pd.read_csv(ARCHIVO_CSV)
    df[COLUMNA_ESTADO] = df[COLUMNA_ESTADO].astype(str).str.strip()
    # Comprobar que la columna de especies existe
    if col_especie not in df.columns:
        print(f"Error: La columna '{col_especie}' no existe en el DataFrame.")
        return

    for var in variables:
        if var not in df.columns:
            print(f"Advertencia: La variable '{var}' no está en el DataFrame. Se omitirá.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # 1. Detectar si es booleana, categórica o si tiene muy pocos valores únicos (ej. 0 y 1)
        es_categorica = (
            pd.api.types.is_bool_dtype(df[var]) or 
            pd.api.types.is_object_dtype(df[var]) or 
            df[var].nunique() <= 5  # Asumimos que si tiene 5 valores o menos, es categórica/ordinal
        )
        
        if es_categorica:
            # GRÁFICO PARA BOOLEANOS / CATEGÓRICOS (Gráfico de barras agrupado)
            sns.countplot(
                data=df, 
                x=col_especie, 
                hue=var, 
                palette='Set2'
            )
            plt.title(f'Frecuencia de {var} (Booleano/Categórico) por Especie', fontsize=14)
            plt.ylabel('Cantidad de Individuos')
            
        else:
            # GRÁFICO PARA CONTINUOS (Boxplot + Puntos superpuestos para ver la distribución real)
            sns.boxplot(
                data=df, 
                x=col_especie, 
                y=var, 
                palette='Set2', 
                showfliers=False # Ocultamos los outliers del boxplot porque los dibujaremos con stripplot
            )
            # Añadimos los puntos reales para ver la densidad y dispersión exacta
            sns.stripplot(
                data=df, 
                x=col_especie, 
                y=var, 
                color='black', 
                alpha=0.4, 
                size=4,
                jitter=True
            )
            plt.title(f'Distribución de {var} (Continuo) por Especie', fontsize=14)
            plt.ylabel(f'Valor de {var}')
            
        # Ajustes estéticos generales
        plt.xlabel('Especie')
        plt.xticks(rotation=45, ha='right') # Rotar las etiquetas si los nombres son largos
        plt.tight_layout() # Evita que se corten los bordes
        
        # Guardar o mostrar (puedes cambiar esto por plt.savefig si prefieres guardarlos en disco)
        plt.show()

if __name__ == "__main__":
    #generar_reportes_modularizados()
    #plotear_varianza_acumulada_grupos()
    #calcular_entropia_combinaciones()

    variables_a_plotear = [
        'Sex_F', 'Sex_indet', 'Teeth_M3', 'Teeth_M2'
    ]

    plot_distribuciones_por_especie(variables_a_plotear, col_especie='Target_Specie')