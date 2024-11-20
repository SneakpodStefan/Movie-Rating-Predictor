import os
from pathlib import Path
from src.data_preparation import load_and_clean_data
from src.data_analysis import analyze_data, print_analysis, analyze_genre_preferences, print_genre_analysis
from src.feature_engineering import create_base_features
from src.model_training import train_host_model, print_model_metrics, setup_logging

def main():
    current_dir = Path(__file__).parent
    DATA_PATH = current_dir / "data" / "raw" / "Sneakpod Punkte.csv"
    
    print("Movie Rating Predictor")
    print("=====================")
    
    if DATA_PATH.exists():
        # Logging Setup
        setup_logging()
        
        # Daten laden und bereinigen
        df = load_and_clean_data(str(DATA_PATH))
        
        # Feature Engineering
        features = create_base_features(df)
        
        # Genre-Analyse
        genre_stats = analyze_genre_preferences(df)
        print_genre_analysis(genre_stats)
        
        # Modelle für jeden Host trainieren
        for host in ['Christoph', 'Robert', 'Stefan']:
            ratings = df[host]
            if len(ratings.dropna()) > 0:
                metrics, model = train_host_model(features, ratings, host)
                print_model_metrics(metrics, host)
    else:
        print(f"Fehler: Keine Daten gefunden unter {DATA_PATH}")
        return

if __name__ == "__main__":
    main()
