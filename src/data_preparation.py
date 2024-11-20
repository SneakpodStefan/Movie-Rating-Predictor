import pandas as pd
import numpy as np

def clean_rating(rating):
    """Bereinigt Bewertungen in ein einheitliches Format"""
    if pd.isna(rating) or rating == '-' or rating == '':
        return np.nan
    try:
        # Ersetze Komma durch Punkt und konvertiere zu float
        return float(str(rating).replace(',', '.'))
    except:
        return np.nan

def load_and_clean_data(file_path):
    """Lädt und bereinigt die Rohdaten"""
    # Lade CSV mit expliziten Spaltenbezeichnungen
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Bereinige Spaltennamen
    df.columns = [col.strip() for col in df.columns]
    
    # Bereinige Bewertungen
    for host in ['Christoph', 'Robert', 'Stefan']:
        df[host] = df[host].apply(clean_rating)
    
    # Bereinige IMDB-Rating
    df['IMDB_Rating'] = df['IMDB_Rating'].apply(clean_rating)
    
    # Konvertiere Jahr zu Integer
    df['Jahr'] = pd.to_numeric(df['Jahr'], errors='coerce')
    
    # Bereinige Genre (von String zu Liste)
    df['Genre'] = df['Genre'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    print(f"Daten geladen und bereinigt:")
    print(f"Anzahl Filme: {len(df)}")
    print("\nBewertungen pro Host:")
    for host in ['Christoph', 'Robert', 'Stefan']:
        valid_ratings = df[host].dropna()
        print(f"{host}: {len(valid_ratings)} Bewertungen, "
              f"Durchschnitt: {valid_ratings.mean():.2f}")
    
    return df
