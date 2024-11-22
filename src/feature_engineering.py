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
    
    # Sprach-Features (als kategorische Variable)
    features['language'] = pd.Categorical(
        df['Sprache'].fillna('Unknown')
    ).codes
    
    # Genre-Features (wichtigste zuerst)
    main_genres = [
        'Drama', 'Comedy', 'Thriller', 'Action', 
        'Crime', 'Adventure', 'Romance', 'Sci-Fi',
        'Horror', 'Biography', 'Animation', 'Family'  # Erweiterte Liste
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
    
    # IMDB Genre-Score
    genre_imdb_means = {}
    for genres in df['Genre']:
        for genre in genres:
            if genre not in genre_imdb_means:
                genre_movies = df[df['Genre'].apply(lambda x: genre in x)]
                genre_imdb_means[genre] = genre_movies['IMDB_Rating'].mean()
    
    # Sicherer Lambda mit Fallback für leere Genre-Listen
    features['genres_imdb_mean'] = df['Genre'].apply(
        lambda genres: sum(genre_imdb_means.get(g, 0) for g in genres) / len(genres) 
        if len(genres) > 0 
        else df['IMDB_Rating'].mean()  # Fallback auf durchschnittliche IMDB-Bewertung
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
    
    # Keyword Features hinzufügen
    keyword_features = create_keyword_features(df)
    
    # Alle Features zusammenführen
    features = pd.concat([features, cast_features, keyword_features], axis=1)
    
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
    features = pd.DataFrame()
    
    df['Regisseur'] = df['Regisseur'].fillna('[]').apply(eval)
    df['Hauptdarsteller'] = df['Hauptdarsteller'].fillna('[]').apply(eval)
    df['Folge'] = df['Folge'].str.extract(r'#(\d+)').astype(float)
    df = df.sort_values('Folge', ascending=False)
    
    def calculate_host_specific_rating(row, person_type, host_name):
        """Berechnet das Rating für einen spezifischen Host und Personentyp"""
        if not row[person_type]:
            return 0
        
        current_folge = row['Folge']
        person_ratings = []
        
        for person in row[person_type]:
            relevant_movies = df[
                (df[person_type].apply(lambda x: isinstance(x, list) and person in x)) &
                (df['Folge'] < current_folge)
            ]
            
            ratings = relevant_movies[host_name].dropna()
            
            if len(ratings) >= 2:
                weights = np.linspace(1, 2, len(ratings))
                avg_rating = np.average(ratings, weights=weights)
                person_ratings.append(avg_rating)
            elif len(ratings) > 0:
                person_ratings.append(ratings.mean())
        
        if person_ratings:
            top_ratings = sorted(person_ratings, reverse=True)[:2]
            return np.mean(top_ratings)
        return 0
    
    # Features für jeden Host erstellen
    for host in ['Christoph', 'Robert', 'Stefan']:
        features[f'{host.lower()}_director_rating'] = df.apply(
            lambda row: calculate_host_specific_rating(row, 'Regisseur', host),
            axis=1
        )
        
        features[f'{host.lower()}_actor_rating'] = df.apply(
            lambda row: calculate_host_specific_rating(row, 'Hauptdarsteller', host),
            axis=1
        )
    
    return features

def create_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt Features aus Plot-Keywords"""
    keyword_features = pd.DataFrame()
    
    # Sammle alle Keywords
    all_keywords = []
    for keywords in df['Plot_Keywords']:
        if isinstance(keywords, list):
            all_keywords.extend([k.lower().strip() for k in keywords])
    
    # Top Keywords nach Häufigkeit
    keyword_counts = Counter(all_keywords)
    
    # Feature-Erstellung für Top 30 Keywords (statt 20)
    top_keywords = [k for k, _ in keyword_counts.most_common(30)]
    
    for keyword in top_keywords:
        col_name = f'keyword_{keyword.replace(" ", "_").replace("-", "_")}'
        keyword_features[col_name] = df['Plot_Keywords'].apply(
            lambda x: 1 if isinstance(x, list) and 
                          any(k.lower().strip() == keyword for k in x) 
            else 0
        )
    
    # Zusätzliche aggregierte Keyword-Features
    keyword_features['keyword_count'] = df['Plot_Keywords'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    
    return keyword_features
