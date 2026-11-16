# KI-gestützte Anomalie- und Ursachenanalyse im Schienenfernverkehr

Eine Python- und Streamlit-Anwendung zur Erkennung von Verspätungsanomalien im ICE-Fernverkehr sowie zur automatisierten Ursachenrecherche mithilfe lokaler Sprachmodelle (Ollama) und externer Websuche (Tavily API).

## Kernfunktionen

* **Datenaufbereitung (`src/preprocessed.py`):** Bereinigung und Strukturierung der historischen Fahrplandaten sowie Export als performante Parquet-Datei (`data/processed/ice_data.parquet`).
* **Interaktive Visualisierung (`src/ui_components.py`):** Explorative Datenanalyse im Streamlit-Dashboard mit Kalender-Heatmaps, Wochentags-Boxplots und Filteroptionen.
* **KI-gestützte Ursachenanalyse (`Ollama` + `Tavily`):** Kombination aus lokalem LLM (`llama3.1`) und gezielter Webrecherche zur Identifikation möglicher Störungsursachen (z. B. Unwetter, Streckensperrungen, Großveranstaltungen) für auffällige Zeiträume.
* **Batch-Evaluation & Feedback-Tracking (`src/run_batch_evaluation.py`, `src/data_manager.py`):** Reproduzierbare Testläufe mit Protokollierung in `evaluation/evaluation_log.csv`, Auswertungsbericht (`evaluation/report.md`) und Speicherung von Nutzerbewertungen (`evaluation/feedback_data.csv`).

## Technologie-Stack

* **Sprache & Datenverarbeitung:** Python 3.12, pandas, NumPy
* **Visualisierung & UI:** Streamlit, Matplotlib, Seaborn
* **LLM & Recherche:** Ollama (`llama3.1`), Tavily Search API

## Projektstruktur

```text
├── data/
│   ├── raw/                     # Unverarbeitete Eingangsdaten (nicht im Repository enthalten)
│   └── processed/               # Aufbereitete Datensätze (ice_data.parquet)
├── evaluation/
│   ├── evaluation_log.csv       # Protokoll der Batch-Evaluation
│   ├── feedback_data.csv        # Gesammeltes Nutzer-Feedback aus dem Dashboard
│   ├── report.md                # Zusammenfassung der Evaluationsergebnisse
│   └── *.png                    # Exportierte Auswertungsdiagramme
├── src/
│   ├── app.py                   # Einstiegspunkt der Streamlit-Anwendung
│   ├── config.py                # Zentrale Konfiguration (Dateipfade, Modellname)
│   ├── data_manager.py          # Datenlade-Logik und Speichern von Nutzer-Feedback
│   ├── preprocessed.py          # Vorverarbeitung der Rohdaten nach Parquet
│   ├── run_batch_evaluation.py  # Skript für automatisierte Evaluationsläufe
│   └── ui_components.py         # Diagramme und UI-Komponenten für Streamlit
├── tests/
│   ├── print_tree.py            # Hilfsskript zur Ausgabe der Verzeichnisstruktur
│   ├── test_num_of_days.py      # Tests zur Datums- und Tagesabdeckung
│   └── test_ollama_setup.py     # Verbindungstest für die lokale Ollama-Instanz
├── .env.example                 # Vorlage für erforderliche Umgebungsvariablen
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation & Setup

1. **Virtuelle Umgebung erstellen und Abhängigkeiten installieren:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Umgebungsvariablen konfigurieren:**
   Erstelle eine `.env`-Datei auf Basis der Vorlage und trage deinen Tavily API-Key ein:
   ```bash
   cp .env.example .env
   ```

3. **Lokales Sprachmodell mit Ollama einrichten:**
   Stelle sicher, dass [Ollama](https://ollama.com/) installiert ist und im Hintergrund läuft, und lade das konfigurierte Modell herunter:
   ```bash
   ollama pull llama3.1
   ```
   Optional kannst du die Erreichbarkeit des Modells vorab testen:
   ```bash
   python tests/test_ollama_setup.py
   ```

## Nutzung

1. **Daten aufbereiten:**
   Lege die Rohdaten im Verzeichnis `data/raw/` ab und führe das Preprocessing aus, um `data/processed/ice_data.parquet` zu erzeugen:
   ```bash
   python src/preprocessed.py
   ```

2. **Streamlit-Dashboard starten:**
   ```bash
   streamlit run src/app.py
   ```

3. **Batch-Evaluation ausführen:**
   ```bash
   python src/run_batch_evaluation.py
   ```
   Die Ergebnisse werden unter `evaluation/evaluation_log.csv` abgelegt.