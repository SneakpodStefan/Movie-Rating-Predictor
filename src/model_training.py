from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def train_host_model(features: pd.DataFrame, 
                    ratings: pd.Series,
                    host: str) -> Tuple[Dict, RandomForestRegressor]:
    """
    Trainiert ein Modell für einen spezifischen Host
    """
    # Entferne Zeilen mit fehlenden Werten
    valid_idx = ratings.dropna().index
    X = features.loc[valid_idx]
    y = ratings[valid_idx]
    
    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    
    # Feature Importance
    importance = dict(zip(X.columns, model.feature_importances_))
    metrics['feature_importance'] = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )
    
    return metrics, model

def print_model_metrics(metrics: Dict, host: str):
    """
    Gibt die Modell-Metriken formatiert aus
    """
    print(f"\nModell-Metriken für {host}")
    print("=" * (18 + len(host)))
    print(f"Trainingsdaten: {metrics['n_train']} Filme")
    print(f"Testdaten: {metrics['n_test']} Filme")
    print("\nFehlermetriken:")
    print(f"Train RMSE: {metrics['train_rmse']:.3f}")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Train MAE: {metrics['train_mae']:.3f}")
    print(f"Test MAE: {metrics['test_mae']:.3f}")
    print(f"Train R²: {metrics['train_r2']:.3f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    
    print("\nTop 10 wichtigste Features:")
    for feature, importance in list(metrics['feature_importance'].items())[:10]:
        print(f"{feature}: {importance:.3f}")
