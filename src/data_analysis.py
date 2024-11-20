import pandas as pd
import numpy as np
from typing import Dict
import logging

def analyze_data(df: pd.DataFrame) -> Dict:
    """
    Führt eine grundlegende Analyse der Filmdaten durch
    """
    analysis = {}
    
    # Grundlegende Statistiken
    analysis['total_movies'] = len(df)
    
    # Bewertungsstatistiken pro Host
    hosts = ['Christoph', 'Robert', 'Stefan']
    host_stats = {}
    for host in hosts:
        ratings = df[host].dropna()
        host_stats[host] = {
            'count': len(ratings),
            'mean': ratings.mean(),
            'median': ratings.median(),
            'std': ratings.std(),
            'min': ratings.min(),
            'max': ratings.max()
        }
    analysis['host_stats'] = host_stats
    
    # IMDB Vergleich
    analysis['imdb_stats'] = {
        'mean': df['IMDB_Rating'].mean(),
        'median': df['IMDB_Rating'].median(),
        'std': df['IMDB_Rating'].std()
    }
    
    # Genre-Analyse
    all_genres = []
    for genres in df['Genre']:
        all_genres.extend(genres)
    genre_counts = pd.Series(all_genres).value_counts()
    analysis['top_genres'] = genre_counts.head(10).to_dict()
    
    # Zeitliche Verteilung
    analysis['year_stats'] = {
        'oldest': df['Jahr'].min(),
        'newest': df['Jahr'].max(),
        'most_common': df['Jahr'].mode()[0]
    }
    
    # Korrelationen zwischen Hosts und IMDB
    correlations = {}
    for host in hosts:
        corr = df[[host, 'IMDB_Rating']].corr().iloc[0,1]
        correlations[f'{host}_imdb_corr'] = corr
    analysis['correlations'] = correlations
    
    return analysis

def print_analysis(df, features):
    # Basis-Statistiken in Konsole
    print("Daten geladen und bereinigt:")
    print(f"Anzahl Filme: {len(df)}\n")
    
    print("Bewertungen pro Host:")
    for host in ['Christoph', 'Robert', 'Stefan']:
        ratings = df[host].dropna()
        print(f"{host}: {len(ratings)} Bewertungen, Durchschnitt: {ratings.mean():.2f}")
    
    print(f"\nFeature Engineering abgeschlossen:")
    print(f"Anzahl Features: {len(features.columns)}")
    
    # Feature-Details nur ins Log schreiben
    logging.info("\nFeature Details:")
    for col in features.columns:
        valid_count = features[col].notna().sum()
        logging.info(f"{col}: {valid_count} gültige Werte")
    
    # Nur kurze Zusammenfassung in Konsole
    print("Feature Engineering erfolgreich abgeschlossen.")
    
    # Bewertungsstatistiken weiterhin in der Konsole zeigen
    print("\nBewertungsstatistiken pro Host:")
    print("--------------------------------")
    for host, stats in df['host_stats'].items():
        print(f"\n{host}:")
        print(f"Anzahl Bewertungen: {stats['count']}")
        print(f"Durchschnitt: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Standardabweichung: {stats['std']:.2f}")
        print(f"Min/Max: {stats['min']:.1f} / {stats['max']:.1f}")
    
    print("\nIMDB Vergleich:")
    print("--------------")
    print(f"Durchschnitt: {df['imdb_stats']['mean']:.2f}")
    print(f"Median: {df['imdb_stats']['median']:.2f}")
    print(f"Standardabweichung: {df['imdb_stats']['std']:.2f}")
    
    print("\nTop 10 Genres:")
    print("-------------")
    for genre, count in df['top_genres'].items():
        print(f"{genre}: {count}")
    
    print("\nZeitliche Verteilung:")
    print("-------------------")
    print(f"Ältester Film: {df['year_stats']['oldest']}")
    print(f"Neuester Film: {df['year_stats']['newest']}")
    print(f"Häufigstes Jahr: {df['year_stats']['most_common']}")
    
    print("\nKorrelationen mit IMDB:")
    print("----------------------")
    for key, corr in df['correlations'].items():
        host = key.split('_')[0]
        print(f"{host}: {corr:.3f}") 

def analyze_genre_preferences(df: pd.DataFrame) -> Dict:
    """Analysiert die Genre-Präferenzen der Hosts im Vergleich zu IMDB"""
    genre_stats = {}
    
    for host in ['Christoph', 'Robert', 'Stefan']:
        host_stats = {}
        host_ratings = df[host].dropna()
        
        # Für jedes Genre
        all_genres = []
        for genres in df['Genre']:
            all_genres.extend(genres)
        unique_genres = list(set(all_genres))
        
        for genre in unique_genres:
            # Filme mit diesem Genre finden
            genre_movies = df[df['Genre'].apply(lambda x: genre in x)]
            genre_ratings = genre_movies[[host, 'IMDB_Rating']].dropna()
            
            if len(genre_ratings) >= 5:  # Mindestens 5 Bewertungen
                # Normalisiere IMDB auf 1-10 Skala
                imdb_mean = genre_ratings['IMDB_Rating'].mean()
                host_mean = genre_ratings[host].mean()
                
                host_stats[genre] = {
                    'count': len(genre_ratings),
                    'host_mean': host_mean,
                    'imdb_mean': imdb_mean,
                    'diff_to_imdb': host_mean - imdb_mean,
                    'ratio_to_imdb': host_mean / imdb_mean
                }
        
        # Nach Abweichung von IMDB sortieren
        host_stats = dict(sorted(
            host_stats.items(),
            key=lambda x: abs(x[1]['diff_to_imdb']),
            reverse=True
        ))
        genre_stats[host] = host_stats
    
    return genre_stats

def print_genre_analysis(genre_stats: Dict):
    print("\nGenre-Präferenzen im Vergleich zu IMDB:")
    print("====================================")
    
    for host, stats in genre_stats.items():
        print(f"\n{host}:")
        print("-" * (len(host) + 1))
        print("Genre (Anzahl) | Host Ø | IMDB Ø | Δ zu IMDB | Faktor")
        print("-" * 60)
        
        for genre, values in stats.items():
            if values['count'] >= 10:  # Nur relevante Genres zeigen
                print(f"{genre:12} ({values['count']:3d}) | "
                      f"{values['host_mean']:6.2f} | "
                      f"{values['imdb_mean']:6.2f} | "
                      f"{values['diff_to_imdb']:+6.2f} | "
                      f"x{values['ratio_to_imdb']:4.2f}")