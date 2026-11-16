import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    print("FEHLER: TAVILY_API_KEY nicht in der .env gefunden!")
else:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    suchbegriff = "Was war die Ursache für die massiven Zugverspätungen und Störungen bei der Deutschen Bahn am 1. Juli 2025?"
    print(f"Suche startet für: '{suchbegriff}'...\n")

    response = tavily.search(
        query=suchbegriff, 
        search_depth="advanced",
        max_results=3
    )

    for i, result in enumerate(response['results'], 1):
        print(f"--- Treffer {i} ---")
        print(f"Titel: {result['title']}")
        print(f"Quelle: {result['url']}")
        print(f"Inhalt: {result['content']}\n")

    print("Suche abgeschlossen!")