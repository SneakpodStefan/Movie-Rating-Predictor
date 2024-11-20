import pandas as pd
import os
import joblib
from pathlib import Path
from .feature_engineering import create_base_features

def predict_ratings(movie_data: dict) -> dict:
    """
    Macht Vorhersagen für einen neuen Film
    
    Args:
        movie_data: Dict mit Film-Informationen:
            - IMDB_Rating: float
            - Runtime: int
            - FSK: str
            - Genre: list[str]
            - Sprache: str
            - Land: str
            - Regisseur: list[str]
            - Hauptdarsteller: list[str]
    
    Returns:
        Dict mit vorhergesagten Bewertungen pro Host
    """
    # DataFrame mit einem Film erstellen
    df = pd.DataFrame([movie_data])
    
    # Features erstellen
    features = create_base_features(df)
    
    # Vorhersagen für jeden Host
    predictions = {}
    current_dir = Path(__file__).parent.parent
    for host in ['Christoph', 'Robert', 'Stefan']:
        model_path = current_dir / 'models' / f'{host}_model.joblib'
        if model_path.exists():
            model = joblib.load(model_path)
            pred = model.predict(features)[0]
            predictions[host] = round(float(pred), 1)
        else:
            print(f"Warnung: Kein Modell gefunden für {host}")
    
    return predictions

def print_prediction(movie: dict):
    """Formatierte Ausgabe der Vorhersagen"""
    predictions = predict_ratings(movie)
    
    print(f"\nVorhersagen für {movie['title']}")
    print("=" * (12 + len(movie['title'])))
    print(f"Genre: {', '.join(movie['Genre'])}")
    print(f"Regie: {', '.join(movie['Regisseur'])}")
    print(f"Cast: {', '.join(movie['Hauptdarsteller'])}")
    print(f"IMDB: {movie['IMDB_Rating']}")
    print("\nVorhergesagte Bewertungen:")
    for host, rating in predictions.items():
        print(f"{host}: {rating}")
