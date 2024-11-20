from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path
from datetime import datetime
import glob
import joblib

def setup_logging():
    # Logging-Verzeichnis erstellen
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Zeitstempel für den Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"model_metrics_{timestamp}.log"
    
    # Alte Logs aufräumen - behalte nur die 10 neuesten
    log_files = sorted(glob.glob(str(log_dir / "model_metrics_*.log")))
    if len(log_files) > 9:  # 9, weil wir gleich eine neue erstellen
        for old_file in log_files[:-9]:
            Path(old_file).unlink()
    
    # Logging konfigurieren
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Root Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Entferne alte Handler
    for old_handler in root_logger.handlers[:]:
        root_logger.removeHandler(old_handler)
    
    # Füge neuen Handler hinzu
    root_logger.addHandler(handler)

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
    
    # Modell auf allen Daten trainieren (für finale Vorhersagen)
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    final_model.fit(features.loc[valid_idx], ratings[valid_idx])
    
    # Modell speichern
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f'{host}_model.joblib'
    joblib.dump(final_model, model_path)
    logging.info(f"\nModell für {host} wurde erstellt und gespeichert unter: {model_path}")
    
    return metrics, final_model

def print_model_metrics(metrics, host):
    # In Datei loggen
    logging.info(f"\nDetaillierte Metriken für {host}")
    logging.info("=" * 50)
    logging.info(f"Anzahl Bewertungen: {metrics['n_train']}")
    logging.info("\nFehlermetriken (5-Fold Cross-Validation):")
    logging.info(f"Train RMSE: {metrics['train_rmse']:.3f}")
    logging.info(f"Test RMSE: {metrics['test_rmse']:.3f}")
    logging.info(f"Train MAE: {metrics['train_mae']:.3f}")
    logging.info(f"Test MAE: {metrics['test_mae']:.3f}")
    logging.info(f"Train R²: {metrics['train_r2']:.3f}")
    logging.info(f"Test R²: {metrics['test_r2']:.3f}")
    logging.info("\nFeature Importance:")
    for feature, importance in metrics['feature_importance'].items():
        logging.info(f"{feature:20} {importance:.3f}")
    
    # Nur kompakte Ausgabe in Konsole
    print(f"\nModell für {host}")
    print("=" * (len(host) + 10))
    print(f"Bewertungen: {metrics['n_train']}")
    
    print("\nPerformance (5-Fold CV):")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Test R²:   {metrics['test_r2']:.3f}")
    
    print("\nWichtigste Features:")
    for feature, importance in list(metrics['feature_importance'].items())[:5]:
        print(f"{feature:20} {importance:.3f}")
    print()
