import sys
from pathlib import Path

# Root-Verzeichnis zum Python-Pfad hinzufügen
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.prediction import print_prediction

test_movies = [
    {
        "title": "Napoleon",
        "IMDB_Rating": 6.7,
        "Runtime": 158,
        "FSK": "12",
        "Genre": ["Action", "Adventure", "Biography", "Drama", "History", "War"],
        "Sprache": "English",
        "Land": "United States",
        "Regisseur": ["Ridley Scott"],
        "Hauptdarsteller": ["Joaquin Phoenix", "Vanessa Kirby"]
    },
    {
        "title": "The Killer",
        "IMDB_Rating": 6.8,
        "Runtime": 118,
        "FSK": "16",
        "Genre": ["Action", "Adventure", "Crime", "Drama", "Thriller"],
        "Sprache": "English",
        "Land": "United States",
        "Regisseur": ["David Fincher"],
        "Hauptdarsteller": ["Michael Fassbender", "Tilda Swinton"]
    },
    {
        "title": "Wonka",
        "IMDB_Rating": 7.3,
        "Runtime": 116,
        "FSK": "0",
        "Genre": ["Adventure", "Comedy", "Family", "Fantasy", "Musical"],
        "Sprache": "English",
        "Land": "United States",
        "Regisseur": ["Paul King"],
        "Hauptdarsteller": ["Timothée Chalamet", "Calah Lane"]
    }
]

if __name__ == "__main__":
    for movie in test_movies:
        print_prediction(movie) 