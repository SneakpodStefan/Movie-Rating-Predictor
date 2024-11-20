from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def train_host_model(features: pd.DataFrame, 
                    ratings: pd.Series,
                    host: str) -> Tuple[Dict, RandomForestRegressor]:
    """
    Trainiert ein Modell für einen spezifischen Host mit 5-fold Cross-Validation
    """
    # Entferne Zeilen mit fehlenden Werten
    valid_idx = ratings.dropna().index
    X = features.loc[valid_idx]
    y = ratings[valid_idx]
    
    # 5-fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metriken für jeden Fold
    fold_metrics = {
        'train_rmse': [],
        'test_rmse': [],
        'train_mae': [],
        'test_mae': [],
        'train_r2': [],
        'test_r2': [],
        'feature_importance': []
    }
    
    # Durchführung der Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Modell trainieren
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Vorhersagen
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metriken berechnen
        fold_metrics['train_rmse'].append(np.sqrt(mean_squared_error(y_train, train_pred)))
        fold_metrics['test_rmse'].append(np.sqrt(mean_squared_error(y_test, test_pred)))
        fold_metrics['train_mae'].append(mean_absolute_error(y_train, train_pred))
        fold_metrics['test_mae'].append(mean_absolute_error(y_test, test_pred))
        fold_metrics['train_r2'].append(r2_score(y_train, train_pred))
        fold_metrics['test_r2'].append(r2_score(y_test, test_pred))
        
        # Feature Importance
        importance = dict(zip(X.columns, model.feature_importances_))
        fold_metrics['feature_importance'].append(importance)
    
    # Durchschnittliche Metriken über alle Folds
    metrics = {
        'train_rmse': np.mean(fold_metrics['train_rmse']),
        'test_rmse': np.mean(fold_metrics['test_rmse']),
        'train_mae': np.mean(fold_metrics['train_mae']),
        'test_mae': np.mean(fold_metrics['test_mae']),
        'train_r2': np.mean(fold_metrics['train_r2']),
        'test_r2': np.mean(fold_metrics['test_r2']),
        'n_train': len(y),
        'fold_metrics': fold_metrics
    }
    
    # Durchschnittliche Feature Importance
    avg_importance = {}
    for feature in X.columns:
        importance_values = [fold[feature] for fold in fold_metrics['feature_importance']]
        avg_importance[feature] = np.mean(importance_values)
    
    metrics['feature_importance'] = dict(
        sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    )
    
    return metrics, model

def print_model_metrics(metrics: Dict, host: str):
    """
    Gibt die Modell-Metriken formatiert aus
    """
    print(f"\nModell-Metriken für {host}")
    print("=" * (18 + len(host)))
    print(f"Anzahl Bewertungen: {metrics['n_train']}")
    
    print("\nDurchschnittliche Fehlermetriken über 5 Folds:")
    print(f"Train RMSE: {metrics['train_rmse']:.3f}")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Train MAE: {metrics['train_mae']:.3f}")
    print(f"Test MAE: {metrics['test_mae']:.3f}")
    print(f"Train R²: {metrics['train_r2']:.3f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    
    print("\nTop 10 wichtigste Features (Durchschnitt über alle Folds):")
    for feature, importance in list(metrics['feature_importance'].items())[:10]:
        print(f"{feature}: {importance:.3f}")
