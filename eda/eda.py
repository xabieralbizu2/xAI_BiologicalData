import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------
# 1. LISTAS DE VARIABLES (Originales Numéricas)
# ---------------------------------------------------------
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

ALL_ORIGINAL_NUMERIC = ISO_VARS + FURROW_VARS + SSFA_VARS

# ---------------------------------------------------------
# 2. FUNCIONES DE ANÁLISIS (RECUPERADAS DEL ORIGINAL)
# ---------------------------------------------------------
def explore_dataset(df, name="Dataset"):
    print(f"\n--- {name} Dimensions ---")
    print("Shape:", df.shape)
    if 'Status' in df.columns:
        print("Distribución por Estado:")
        print(df['Status'].value_counts())
    if 'Target_Specie' in df.columns:
        print(f"Distribución de Especies en {name}:")
        print(df['Target_Specie'].value_counts())

def internal_correlations(df, name, plot=False):
    """Calcula y plotea la distribución de correlaciones internas de un bloque."""
    if df.empty or df.shape[1] < 2:
        return
    
    corr_matrix = df.corr()
    # Triángulo superior sin diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    int_corr_vals = corr_matrix.where(mask).stack().values
    
    mean_corr = np.nanmean(np.abs(int_corr_vals)) if len(int_corr_vals) > 0 else 0
    print(f"Correlación media (abs) en {name}: {mean_corr:.2f}")

    if plot and len(int_corr_vals) > 0:
        sns.kdeplot(int_corr_vals, label=name, fill=True, alpha=0.3)

def std_skew(dfs, names):
    """Genera boxplots de dispersión relativa (std/mean) y asimetría (skew)."""
    blocks = []
    for (df, name) in zip(dfs, names):
        if df.empty: continue
        
        stats = df.describe().T
        mean_safe = stats["mean"].replace(0, np.nan)
        
        summary = pd.DataFrame({
            "std_mean_ratio": stats["std"] / mean_safe.abs(),
            "skew": df.skew(numeric_only=True),
            "range": stats["max"] - stats["min"]
        })
        summary["block"] = name
        blocks.append(summary)
    
    if not blocks:
        return

    summary = pd.concat(blocks)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x="block", y="std_mean_ratio", data=summary, ax=axes[0])
    axes[0].set_title("Dispersión relativa (std/mean)")
    sns.boxplot(x="block", y="skew", data=summary, ax=axes[1])
    axes[1].set_title("Asimetría (skew)")
    plt.tight_layout()
    plt.show()

def corr_above_th(df, thresholds):
    """Plotea el porcentaje de pares de variables con correlación superior a X."""
    corr = df.corr(method='spearman').abs()
    percentages = []
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper_vals = corr.where(mask).stack()
    total_pairs = len(upper_vals)

    if total_pairs == 0:
        return

    for threshold in thresholds:
        count = (upper_vals > threshold).sum()
        percentages.append(100 * count / total_pairs)

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, percentages, marker='o', linestyle='-', color='purple')
    plt.xlabel('Threshold correlation (|r|)')
    plt.ylabel('% of pairs > threshold')
    plt.title('Redundancia Global (Spearman)')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_pca_analysis(df, numeric_cols, label_col, title="PCA"):
    """PCA 2D coloreado por la columna label_col."""
    cols_exist = [c for c in numeric_cols if c in df.columns]
    X = df[cols_exist].fillna(0)
    
    if len(cols_exist) < 2 or X.shape[0] < 2:
        print("⚠️ No hay suficientes datos/columnas para PCA.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['Label'] = df[label_col].values
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Label', data=pca_df, alpha=0.7, s=80, palette='viridis')
    plt.title(f"{title} (Var Explicada: {pca.explained_variance_ratio_.sum():.2%})")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def feature_importance_mi(df, numeric_cols, target_col):
    """Calcula Mutual Information para ver qué variables importan más."""
    cols_exist = [c for c in numeric_cols if c in df.columns]
    X = df[cols_exist].fillna(0)
    y = df[target_col]
    
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=cols_exist).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    mi_series.head(15).plot(kind='barh', color='teal')
    plt.title(f"Top 15 Variables (Mutual Info) - {target_col}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def remove_high_corr_and_plot(df, numeric_cols, threshold=0.9):
    """
    Elimina variables con correlación absoluta mayor al threshold y 
    grafica la matriz de correlación resultante.
    """
    print(f"\n--- Eliminación de variables correlacionadas (Th={threshold}) ---")
    
    # Filtrar solo columnas presentes
    valid_cols = [c for c in numeric_cols if c in df.columns]
    df_num = df[valid_cols]
    
    # Calcular matriz de correlación absoluta
    corr_matrix = df_num.corr().abs()
    
    # Seleccionar el triángulo superior de la matriz
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identificar columnas a eliminar (cualquiera que tenga > threshold en el triángulo superior)
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Variables eliminadas ({len(to_drop)}):")
    print(to_drop)
    
    # Crear DF limpio
    df_clean = df.drop(columns=to_drop)
    
    # Calcular nueva matriz de correlación para graficar
    # (Usamos las columnas numéricas que quedaron)
    remaining_cols = [c for c in valid_cols if c not in to_drop]
    new_corr = df_clean[remaining_cols].corr()
    
    # Plotear Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(new_corr, cmap='RdBu_r', center=0, square=True, vmax=1, vmin=-1)
    plt.title(f"Matriz de Correlación tras limpiar (Th={threshold})\nVariables restantes: {len(remaining_cols)}")
    plt.tight_layout()
    plt.show()
    
    return df_clean

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. CARGA DE DATOS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE = os.path.join(BASE_DIR, "src/dataset_complete_onehot.csv")

    df = pd.read_csv(FILE)

    STATUS_COL = 'Status'
    TARGET_COL = 'Specie'

    # 2. DEFINIR VARIABLES
    # Detectar One-Hot dinámicamente
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    one_hot_vars = [c for c in all_numeric_cols if c not in ALL_ORIGINAL_NUMERIC]
    full_features = [c for c in ALL_ORIGINAL_NUMERIC if c in df.columns] + one_hot_vars

    # 3. ANÁLISIS COMPARATIVO (EXTANT vs EXTINCT)
    print("\n" + "="*40)
    print(" 1. COMPARATIVA GLOBAL: VIVOS vs FÓSILES")
    print("="*40)
    explore_dataset(df, "Dataset Completo")
    
    # PCA Global para ver overlap
    plot_pca_analysis(df, full_features, STATUS_COL, title="PCA: Extant vs Extinct")

    # 4. ANÁLISIS PROFUNDO DE FÓSILES (RESTAURADO)
    print("\n" + "="*40)
    print(" 2. ANÁLISIS ESTADÍSTICO DE FÓSILES (EXTINCT)")
    print("="*40)
    
    df_extinct = df[df[STATUS_COL] == 'Extinct'].copy()
    
    if df_extinct.empty:
        print("⚠️ No hay datos Extinct para analizar.")
    else:
        explore_dataset(df_extinct, "Subset Extinct")

        # Preparar sub-dataframes para el análisis por bloques (ISO, Furrow, SSFA)
        # Esto es lo que hacía tu script original
        def safe_slice(df, cols):
            valid = [c for c in cols if c in df.columns]
            return df[valid]

        df_iso = safe_slice(df_extinct, ISO_VARS)
        df_furrow = safe_slice(df_extinct, FURROW_VARS)
        df_ssfa = safe_slice(df_extinct, SSFA_VARS)
        
        # A. Correlaciones Internas (KDE Plot)
        print("\n>>> A. Correlaciones Internas por bloque")
        plt.figure(figsize=(10, 6))
        internal_correlations(df_iso, "ISO", plot=False) # Plotear todo junto a veces ensucia, mejor solo stats o uno limpio
        internal_correlations(df_furrow, "Furrow", plot=False)
        internal_correlations(df_ssfa, "SSFA", plot=True) # Ploteamos SSFA como ejemplo o todos si prefieres
        plt.title("Distribución de Correlaciones Internas (Extinct)")
        plt.xlabel("|Correlación Pearson|")
        plt.show()

        # B. Skewness y Dispersión (Boxplots)
        print("\n>>> B. Calidad del Dato (Skew & Std/Mean)")
        std_skew([df_iso, df_furrow, df_ssfa], ["ISO", "Furrow", "SSFA"])

        # C. Redundancia Global (Line Plot)
        print("\n>>> C. Redundancia Global (Umbrales)")
        # Unimos solo las variables originales numéricas para este test
        df_numeric_orig = pd.concat([df_iso, df_furrow, df_ssfa], axis=1)
        corr_above_th(df_numeric_orig, thresholds=[0.5, 0.7, 0.8, 0.9, 0.95])

        # D. PCA Específico (Por Especie)
        print("\n>>> D. PCA de Especies Extintas")
        # Aquí usamos también las variables One-Hot (sexo, diente) porque ayudan a separar
        plot_pca_analysis(df_extinct, full_features, TARGET_COL, title="PCA: Especies Extintas")

        # E. Importancia de Variables
        print("\n>>> E. Variables Discriminantes (Mutual Info)")
        feature_importance_mi(df_extinct, full_features, TARGET_COL)

        print("\n>>> F. Limpieza de Correlaciones y Heatmap Final")
        df_clean = remove_high_corr_and_plot(df, full_features, threshold=0.95)

    print("\n✅ Script finalizado exitosamente.")