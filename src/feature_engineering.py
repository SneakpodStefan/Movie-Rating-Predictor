import pandas as pd
import numpy as np
from typing import List, Dict

def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt Feature-Matrix aus dem Eingabe-DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame mit Rohdaten
        
    Returns:
        pd.DataFrame: Feature-Matrix mit engineerten Features
    """
    """Erstellt grundlegende Features aus den Rohdaten"""
    features = pd.DataFrame()
    
    # IMDB Features
    features['imdb_rating'] = df['IMDB_Rating']
    
    # Zeitliche Features
    current_year = 2024
    features['movie_age'] = current_year - df['Jahr']
    features['is_recent'] = (current_year - df['Jahr'] <= 2).astype(int)
    
    # Genre Features
    main_genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Crime', 
                  'Adventure', 'Romance', 'Sci-Fi', 'Mystery', 'Fantasy']
    
    for genre in main_genres:
        features[f'is_{genre.lower()}'] = df['Genre'].apply(
            lambda x: int(genre in x)
        )
    
    # Genre-Kombinationen
    features['genre_count'] = df['Genre'].apply(len)
    features['is_drama_comedy'] = features['is_drama'] & features['is_comedy']
    features['is_action_thriller'] = features['is_action'] & features['is_thriller']
    
    # Laufzeit-Features
    features['runtime'] = df['Runtime']
    features['is_long_movie'] = (df['Runtime'] > 120).astype(int)
    
    # FSK Features (als Kategorien)
    features['fsk'] = pd.Categorical(df['FSK']).codes
    
    # Sprach-Features
    features['is_english'] = (df['Sprache'] == 'English').astype(int)
    features['is_german'] = (df['Sprache'] == 'German').astype(int)
    
    print("\nFeature Engineering abgeschlossen:")
    print(f"Anzahl Features: {len(features.columns)}")
    print("\nErstellte Features:")
    for col in features.columns:
        non_null = features[col].count()
        print(f"{col}: {non_null} gültige Werte")
    
    return features

def analyze_feature_importance(features: pd.DataFrame, target: pd.Series, 
                             feature_names: List[str]) -> Dict[str, float]:
    """Analysiert die Wichtigkeit der Features für die Zielvariable"""
    importance = {}
    
    for feature in feature_names:
        if feature in features.columns:
            correlation = features[feature].corr(target)
            importance[feature] = abs(correlation)
    
    # Sortiere nach absoluter Korrelation
    importance = dict(sorted(importance.items(), 
                           key=lambda x: abs(x[1]), 
                           reverse=True))
    
    return importance
