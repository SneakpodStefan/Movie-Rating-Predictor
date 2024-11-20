import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from collections import Counter

import numpy as np
from typing import List, Dict

def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt optimierte Features aus den Rohdaten"""
    features = pd.DataFrame()
    
    # Wichtigste Basis-Features
    features['imdb_rating'] = df['IMDB_Rating']
    features['runtime'] = df['Runtime']
    features['fsk'] = pd.Categorical(df['FSK']).codes
    
    # Genre-Features (wichtigste zuerst)
    main_genres = [
        'Drama', 'Comedy', 'Thriller', 'Action', 
        'Crime', 'Adventure', 'Romance', 'Sci-Fi'
    ]
    
    for genre in main_genres:
        features[f'is_{genre.lower()}'] = df['Genre'].apply(
            lambda x: int(genre in x)
        )
    
    # Sinnvolle Genre-Kombinationen
    features['genre_count'] = df['Genre'].apply(len)
    features['is_drama_comedy'] = features['is_drama'] & features['is_comedy']
    features['is_action_thriller'] = features['is_action'] & features['is_thriller']
    features['is_drama_romance'] = features['is_drama'] & features['is_romance']
    features['is_scifi_action'] = features['is_sci-fi'] & features['is_action']
    
    # Sprach-Features
    features['is_english'] = (df['Sprache'] == 'English').astype(int)
    features['is_german'] = (df['Sprache'] == 'German').astype(int)
    
    # Gewichtete Genre-Scores basierend auf Host-Präferenzen
    features['action_adventure_score'] = (
        features['is_action'] * 0.5 + 
        features['is_adventure'] * 0.5
    )
    
    features['drama_romance_score'] = (
        features['is_drama'] * 0.6 + 
        features['is_romance'] * 0.4
    )
    
    # Runtime-basierte Features
    features['is_long_movie'] = (df['Runtime'] > 120).astype(int)
    features['is_short_movie'] = (df['Runtime'] < 90).astype(int)
    
    # Länder One-Hot-Encoding - alle Länder berücksichtigen
    all_countries = df['Land'].unique()
    
    for country in all_countries:
        if pd.notna(country):  # Nur gültige Länder
            features[f'country_{country.lower().replace(" ", "_")}'] = (
                df['Land'] == country
            ).astype(int)
    
    # Fehlende Werte als eigene Kategorie
    features['country_unknown'] = df['Land'].isna().astype(int)
    
    # Cast Features hinzufügen
    cast_features = create_cast_features(df)
    features = pd.concat([features, cast_features], axis=1)
    
    # Nur Zusammenfassung in Konsole
    print(f"\nFeature Engineering abgeschlossen:")
    print(f"Anzahl Features: {len(features.columns)}")
    
    # Details ins Log
    logging.info("\nErstellte Features:")
    for col in features.columns:
        valid_count = features[col].notna().sum()
        logging.info(f"{col}: {valid_count} gültige Werte")
    
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

def create_cast_features(df):
    """Erstellt aggregierte Cast-Features"""
    features = pd.DataFrame()
    
    # Listen extrahieren
    df['Regisseur'] = df['Regisseur'].fillna('[]').apply(eval)
    df['Hauptdarsteller'] = df['Hauptdarsteller'].fillna('[]').apply(eval)
    
    # Durchschnittliche Bewertung pro Film für jeden Host
    host_ratings = {}
    for host in ['Christoph', 'Robert', 'Stefan']:
        host_ratings[host] = {}
        for idx, row in df.iterrows():
            if pd.notna(row[host]):
                host_ratings[host][idx] = row[host]
    
    def calculate_avg_rating(row, person_type):
        if not row[person_type]:  # Leere Liste
            return 0
        
        all_ratings = []
        for host in host_ratings:
            person_ratings = []
            # Finde alle Filme mit dieser Person
            relevant_indices = df[
                df[person_type].apply(lambda x: any(p in x for p in row[person_type]))
            ].index
            
            # Sammle alle Bewertungen für diese Filme
            for idx in relevant_indices:
                if idx in host_ratings[host]:
                    person_ratings.append(host_ratings[host][idx])
            
            if person_ratings:  # Nur wenn Bewertungen vorhanden
                all_ratings.append(np.mean(person_ratings))
        
        return np.mean(all_ratings) if all_ratings else 0
    
    # Aggregierte Features
    features['director_avg_rating'] = df.apply(
        lambda row: calculate_avg_rating(row, 'Regisseur'),
        axis=1
    )
    
    features['actor_avg_rating'] = df.apply(
        lambda row: calculate_avg_rating(row, 'Hauptdarsteller'),
        axis=1
    )
    
    # Häufigkeits-Features
    features['frequent_director'] = df['Regisseur'].apply(
        lambda x: int(len(x) > 0 and any(
            len(df[df['Regisseur'].apply(lambda y: d in y)]) >= 3 
            for d in x
        ))
    )
    
    features['frequent_actor'] = df['Hauptdarsteller'].apply(
        lambda x: int(len(x) > 0 and any(
            len(df[df['Hauptdarsteller'].apply(lambda y: a in y)]) >= 3
            for a in x
        ))
    )
    
    return features
