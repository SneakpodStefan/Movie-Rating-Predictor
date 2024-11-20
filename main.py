import os
from pathlib import Path
from src.data_preparation import load_and_clean_data

def main():
    # Finde den Basis-Pfad des Projekts
    current_dir = Path(__file__).parent
    DATA_PATH = current_dir / "data" / "raw" / "Sneakpod Punkte.csv"
    
    print("Movie Rating Predictor")
    print("=====================")
    print(f"Suche Daten unter: {DATA_PATH}")
    
    # Daten laden und bereinigen
    if DATA_PATH.exists():
        df = load_and_clean_data(str(DATA_PATH))
        print("\nDatei erfolgreich geladen!")
    else:
        print(f"Fehler: Keine Daten gefunden unter {DATA_PATH}")
        print("\nVerfügbare Dateien in data/raw:")
        raw_dir = current_dir / "data" / "raw"
        if raw_dir.exists():
            print(list(raw_dir.glob("*")))
        return

if __name__ == "__main__":
    main()
