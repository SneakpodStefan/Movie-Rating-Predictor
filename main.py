import os
from pathlib import Path
from src.data_preparation import load_and_clean_data
from src.data_analysis import analyze_data, print_analysis

def main():
    # Finde den Basis-Pfad des Projekts
    current_dir = Path(__file__).parent
    DATA_PATH = current_dir / "data" / "raw" / "Sneakpod Punkte.csv"
    
    print("Movie Rating Predictor")
    print("=====================")
    
    # Daten laden und bereinigen
    if DATA_PATH.exists():
        df = load_and_clean_data(str(DATA_PATH))
        
        # Datenanalyse durchführen
        analysis = analyze_data(df)
        print_analysis(analysis)
    else:
        print(f"Fehler: Keine Daten gefunden unter {DATA_PATH}")
        return

if __name__ == "__main__":
    main()
