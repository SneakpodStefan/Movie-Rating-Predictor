import pandas as pd
import numpy as np
from typing import Dict
import logging
from collections import Counter

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
    
    # Keyword-Analyse
    keyword_analysis = analyze_keywords(df)
    analysis['keyword_stats'] = keyword_analysis
    
    return analysis

def print_analysis(df: pd.DataFrame, features: pd.DataFrame, analysis: Dict):
    print("\nBewertungsstatistiken:")
    print("=====================")
    for host in ['Christoph', 'Robert', 'Stefan']:
        ratings = df[host].dropna()
        print(f"{host}: {len(ratings)} Bewertungen, Ø {ratings.mean():.2f}")
    
    # IMDB Vergleich
    print(f"\nIMDB: Ø {analysis['imdb_stats']['mean']:.2f}")
    
    # Rest ins Log schreiben
    logging.info("\nDetaillierte Statistiken:")
    logging.info("======================")
    logging.info(f"\nTop Genres:")
    for genre, count in analysis['top_genres'].items():
        logging.info(f"{genre:12} {count:4d}")
    
    logging.info(f"\nZeitliche Verteilung:")
    logging.info(f"Ältester Film: {analysis['year_stats']['oldest']}")
    logging.info(f"Neuester Film: {analysis['year_stats']['newest']}")

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
    # Nur ins Log schreiben
    logging.info("\nDetaillierte Genre-Analyse:")
    logging.info("=========================")
    for host, stats in genre_stats.items():
        logging.info(f"\n{host}:")
        for genre, values in stats.items():
            if values['count'] >= 10:
                logging.info(f"{genre:12} ({values['count']:3d}) | "
                           f"{values['host_mean']:6.2f} | "
                           f"{values['imdb_mean']:6.2f} | "
                           f"{values['diff_to_imdb']:+6.2f}")

def analyze_keywords(df: pd.DataFrame) -> Dict:
    """Analysiert die häufigsten Keywords und ihre Bewertungskorrelationen"""
    print("\nStarting keyword analysis...")
    
    keyword_stats = {}
    hosts = ['Christoph', 'Robert', 'Stefan']
    
    # Host-Durchschnitte vorab berechnen
    host_means = {
        host: df[host].mean() 
        for host in hosts
    }
    
    # Für jedes Keyword Statistiken berechnen
    for idx, row in df.iterrows():
        keywords = row['Plot_Keywords']
        if isinstance(keywords, list):
            for keyword in keywords:
                keyword = keyword.lower().strip()
                if keyword not in keyword_stats:
                    keyword_stats[keyword] = {
                        'count': 0,
                        'host_stats': {
                            host: {
                                'sum': 0, 
                                'count': 0,
                                'mean': 0,
                                'diff_to_overall': 0
                            } for host in hosts
                        }
                    }
                
                keyword_stats[keyword]['count'] += 1
                
                # Host-Bewertungen sammeln
                for host in hosts:
                    if pd.notna(row[host]):
                        stats = keyword_stats[keyword]['host_stats'][host]
                        stats['sum'] += row[host]
                        stats['count'] += 1
                        if stats['count'] > 0:
                            stats['mean'] = stats['sum'] / stats['count']
                            stats['diff_to_overall'] = stats['mean'] - host_means[host]
    
    # Top Keywords ausgeben
    print(f"\nAnzahl unterschiedlicher Keywords: {len(keyword_stats)}")
    print("\nTop 10 häufigste Keywords:")
    sorted_keywords = sorted(keyword_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for keyword, stats in sorted_keywords[:10]:
        print(f"{keyword}: {stats['count']}")
    
    return keyword_stats

def print_keyword_analysis(keyword_stats: Dict):
    # Log statt Print für die detaillierte Keyword-Analyse
    logging.info("\nDetaillierte Keyword-Analyse:")
    logging.info("============================")
    
    # Sortiere Keywords nach Häufigkeit
    sorted_keywords = sorted(
        keyword_stats.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Schreibe alle Keywords ins Log
    for keyword, stats in sorted_keywords:
        if stats['count'] >= 20:  # Mindestens 20 Filme
            logging.info(f"\n{keyword} ({stats['count']} Filme):")
            logging.info("-" * (len(keyword) + len(str(stats['count'])) + 9))
            
            for host, host_stat in stats['host_stats'].items():
                if host_stat['count'] > 0:
                    diff = host_stat['diff_to_overall']
                    direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                    logging.info(f"{host}: {host_stat['mean']:.2f} "
                               f"({direction} {abs(diff):.2f}) "
                               f"aus {host_stat['count']} Bewertungen")
    
    # Nur eine Zusammenfassung in der Konsole
    print(f"\nKeyword-Analyse: {len(keyword_stats)} verschiedene Keywords gefunden")
    print(f"Details wurden ins Log geschrieben")
