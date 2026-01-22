#----LogReg----
C_values_logreg = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
param_grid_logreg = [
    # --- MODELO 1: L2 ---
    {
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'multi_class': ['multinomial'],
        'C': C_values_logreg
    },
    # --- MODELO 2: L1 ---
    {
        'penalty': ['l1'],
        'solver': ['liblinear'], # Único compatible con L1 en problemas pequeños/medianos
        'multi_class': ['ovr'], # Único compatible con liblinear
        'C': C_values_logreg
    },
    # --- MODELO 3: Elastic Net ---
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'], # Único compatible con elasticnet y multinomial
        'multi_class': ['multinomial'],
        'l1_ratio': [0.0 ,0.1, 0.5, 0.9, 0.95], # Proporción de la mezcla L1/L2
        'C': C_values_logreg
    }
]

param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None], # None significa sin límite
    
    'min_samples_split': [7, 10, 13], 
    
    'min_samples_leaf': [1, 3, 5],
    
    'criterion': ['gini', 'entropy']
}


param_grid_mlp = {
    'hidden_layer_sizes': [
        (3,), (5,), (7,), (10,), (15,), (20,), 
        (3, 3), (10, 10)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.15, 0.2, 0.25, 0.3],
    'learning_rate': ['constant', 'adaptive'],
}

#----Naive Bayes----
import numpy as np
param_grid_nb = {
    'var_smoothing': np.logspace(0, -14, num=100)
}

#----LDA (Linear Discriminant Analysis)----
param_grid_lda = [
    # --- Configuración 1: Solver SVD (Sin Shrinkage, por defecto) ---
    {
        'solver': ['svd'],
    },
    # --- Configuración 2: Shrinkage ---
    {
        'solver': ['lsqr'], 
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
    }
]

#----XGBoost----
param_grid_xgb = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005], 
    'max_depth': [3, 5, 7], 
    'min_child_weight': [1, 3, 5], 
    'gamma': [0.001, 0.002, 0.003, 0.04, 0.05],
}