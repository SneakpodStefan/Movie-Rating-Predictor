import pandas as pd
import numpy as np
from typing import Dict

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

def print_analysis(analysis: Dict):
    """
    Gibt die Analyseergebnisse formatiert aus
    """
    print("\nFilmanalyse")
    print("===========")
    print(f"\nGesamtzahl Filme: {analysis['total_movies']}")
    
    print("\nBewertungsstatistiken pro Host:")
    print("--------------------------------")
    for host, stats in analysis['host_stats'].items():
        print(f"\n{host}:")
        print(f"Anzahl Bewertungen: {stats['count']}")
        print(f"Durchschnitt: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Standardabweichung: {stats['std']:.2f}")
        print(f"Min/Max: {stats['min']:.1f} / {stats['max']:.1f}")
    
    print("\nIMDB Vergleich:")
    print("--------------")
    print(f"Durchschnitt: {analysis['imdb_stats']['mean']:.2f}")
    print(f"Median: {analysis['imdb_stats']['median']:.2f}")
    print(f"Standardabweichung: {analysis['imdb_stats']['std']:.2f}")
    
    print("\nTop 10 Genres:")
    print("-------------")
    for genre, count in analysis['top_genres'].items():
        print(f"{genre}: {count}")
    
    print("\nZeitliche Verteilung:")
    print("-------------------")
    print(f"Ältester Film: {analysis['year_stats']['oldest']}")
    print(f"Neuester Film: {analysis['year_stats']['newest']}")
    print(f"Häufigstes Jahr: {analysis['year_stats']['most_common']}")
    
    print("\nKorrelationen mit IMDB:")
    print("----------------------")
    for key, corr in analysis['correlations'].items():
        host = key.split('_')[0]
        print(f"{host}: {corr:.3f}") 