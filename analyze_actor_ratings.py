import pandas as pd
from pathlib import Path

def analyze_movie_actors():
    # Lade Daten
    current_dir = Path(__file__).parent
    DATA_PATH = current_dir / "data" / "raw" / "Sneakpod Punkte.csv"
    df = pd.read_csv(DATA_PATH)

    # Finde Filme mit Bewertungen von allen drei Hosts
    complete_ratings = df[df['Christoph'].notna() & 
                        df['Robert'].notna() & 
                        df['Stefan'].notna()]
    
    print(f"\nGefundene Filme mit Bewertungen von allen Hosts: {len(complete_ratings)}")
    print("\nDie letzten 5 dieser Filme:")
    for i, movie in complete_ratings.tail().iterrows():
        print(f"{movie['Filmtitel']}: C={movie['Christoph']} R={movie['Robert']} S={movie['Stefan']}")
    
    # Nutzer wählt einen Film
    print("\nWelchen Film möchtest du analysieren? (1-5):")
    choice = int(input()) - 1
    example_movie = complete_ratings.iloc[-(5-choice)]
    
    print(f"\nAnalyse für Film: {example_movie['Filmtitel']}")
    print("-" * (len(example_movie['Filmtitel']) + 16))
    print(f"Schauspieler: {example_movie['Hauptdarsteller']}")
    print("\nBewertungen dieses Films:")
    print(f"Christoph: {example_movie['Christoph']}")
    print(f"Robert: {example_movie['Robert']}")
    print(f"Stefan: {example_movie['Stefan']}")

    # Finde alle Filme mit diesen Schauspielern
    try:
        actors = eval(example_movie['Hauptdarsteller'])
        for actor in actors:
            print(f"\nFilme mit {actor}:")
            print("-" * (len(actor) + 10))
            actor_movies = df[df['Hauptdarsteller'].apply(lambda x: isinstance(x, str) and actor in x)]
            
            # Sammle alle Bewertungen
            ratings = {'Christoph': [], 'Robert': [], 'Stefan': []}
            for _, movie in actor_movies.iterrows():
                print(f"- {movie['Filmtitel']}")
                for host in ['Christoph', 'Robert', 'Stefan']:
                    if pd.notna(movie[host]):
                        ratings[host].append(movie[host])
                        print(f"  {host}: {movie[host]}")
            
            # Berechne Durchschnitte
            print(f"\nDurchschnittliche Bewertungen für {actor}:")
            for host, host_ratings in ratings.items():
                if host_ratings:
                    avg = sum(host_ratings) / len(host_ratings)
                    print(f"{host}: {avg:.2f} (aus {len(host_ratings)} Bewertungen)")

    except Exception as e:
        print(f"Fehler bei der Analyse: {str(e)}")

if __name__ == "__main__":
    analyze_movie_actors() 