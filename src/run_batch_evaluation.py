"""Modul zur automatisierten Evaluation der Pipeline.
Dieses Skript steuert den seriellen Durchlauf über alle definierten
Anomalietage, misst die Laufzeiten der einzelnen Pipeline-Schritte und
sichert sowohl die qualitativen Berichte als auch die quantitativen Metriken.
"""

import pandas as pd
import time
import csv
import os
from datetime import datetime
from reasoning_engine import ReasoningEngine
import data_manager as dm
import streamlit as st
import shutil


ANOMALY_DATES = [
    "2024-08-09", "2024-09-07", "2024-11-14", "2025-06-30", "2025-07-01",
    "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-10", "2025-07-14",
    "2025-09-19", "2025-09-22", "2025-10-13", "2025-10-15", "2025-10-23",
    "2025-10-24", "2025-10-30", "2026-01-12", "2026-01-26", "2026-01-27",
    "2026-01-30"
]

REPORT_FILE = "report.md"
LOG_FILE = "evaluation/evaluation_log.csv"
MODEL_NAME = "command-r"  
CONTEXT_WINDOW = 8192     

def run_batch() -> None:
    """Führt die automatisierte Batch-Evaluation für alle Anomalietage aus.
    Die Funktion bereinigt initial alle Caches, lädt die Datenbasis,
    initialisiert die ReasoningEngine und durchläuft sequentiell alle Testtage.
    Fehler innerhalb eines Tageslaufs werden isoliert abgefangen.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".streamlit", "cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("Cache gelöscht.")
        except Exception as e:
            print(f"Cache konnte nicht gelöscht werden: {e}")
            
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("Cache gelöscht.")
    except Exception:
        pass

    full_df, daily_data, threshold_val, avg_delay_global, _, context_data, _ = dm.load_all_data()
    daily_data['datum'] = pd.to_datetime(daily_data['datum'])
    engine = ReasoningEngine(model_name=MODEL_NAME, validator_model='llama3.1')
    
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
        
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as lf:
            writer = csv.writer(lf)
            writer.writerow([
                "datum", "start_zeit", "dauer_gesamt_sek", 
                "dauer_suche_sek", "dauer_filter_sek", "dauer_ki_sek", 
                "verspaetung_min", "quellen_anzahl", "status"
            ])

    print(f"=== Skript Start: Automatisierte Analyse ===")
    print(f"-> Modell: {MODEL_NAME} | Kontext: {CONTEXT_WINDOW} Token | Tage: {len(ANOMALY_DATES)}")
    print("-" * 60)
    
    for i, date_str in enumerate(ANOMALY_DATES):
        start_time = time.time()
        start_ts = datetime.now().strftime("%H:%M:%S")
        
        try:
            day_ts = pd.Timestamp(date_str)
            day_row = daily_data[daily_data['datum'] == day_ts]
            
            if day_row.empty:
                print(f"[{i+1}/{len(ANOMALY_DATES)}] {date_str}: Keine Daten in der CSV gefunden. Überspringe...")
                continue
                
            delay = day_row['delay_in_min'].values[0]

            def get_clean_ctx(ts: pd.Timestamp) -> dict:
                """Hilfsfunktion zur Extraktion von Z-Score und Ereignistyp."""
                d_str = ts.strftime("%Y-%m-%d")
                c = context_data.get(d_str, {})
                return {
                    "severity_z_score": c.get('severity_z_score', 0), 
                    "event_type": c.get('event_type', 'Normal')
                }
            
            prev_day_ts = day_ts - pd.Timedelta(days=1)
            next_day_ts = day_ts + pd.Timedelta(days=1)
            temporal_context = {
                "prev": get_clean_ctx(prev_day_ts), 
                "next": get_clean_ctx(next_day_ts)
            }

            print(f"[{i+1}/{len(ANOMALY_DATES)}] Pipeline gestartet für {date_str}...")
            news_results = engine.search_news_for_date(date_str)
            
            result_data = engine.generate_explanation(
                date_str=date_str, 
                delay_value=delay, 
                avg_delay_global=avg_delay_global, 
                threshold_val=threshold_val, 
                news_results=news_results,
                context_size=CONTEXT_WINDOW,
                temporal_context=temporal_context
            )
            
            duration_total = round(time.time() - start_time, 2)
            timings = result_data.get('timings', {})
            dur_search = timings.get('search', 0.0)
            dur_filter = timings.get('filtering', 0.0)
            dur_ki = timings.get('final_analysis', 0.0)
            
            sources_count = len(result_data.get('sources', []))
            
            write_markdown(date_str, delay, result_data)
            log_result(date_str, start_ts, duration_total, dur_search, dur_filter, dur_ki, delay, sources_count, "Erfolg")
            
            print(f" Fertig! {duration_total}s (Suche: {dur_search}s | Filter: {dur_filter}s | KI: {dur_ki}s) | Quellen: {sources_count}")
            print("-" * 60)
            time.sleep(2)

        except Exception as e:
            duration_total = round(time.time() - start_time, 2)
            log_result(date_str, start_ts, duration_total, 0.0, 0.0, 0.0, 0.0, 0, f"Fehler: {str(e)}")
            print(f"Fehler bei {date_str}: {e}")
            print("-" * 60)

    print(f"\n=== Prozess beendet ===")
    print(f"-> Berichte gespeichert in: {REPORT_FILE}")
    print(f"-> Laufzeit-Statistiken in: {LOG_FILE}")

def log_result(date: str, start: str, dur_total: float, dur_search: float, 
               dur_filter: float, dur_ki: float, delay: float, sources: int, status: str) -> None:
    """Schreibt die gemessenen Laufzeitmetriken zeilenweise in das CSV-Log.
    
    Args:
        date (str): Der bearbeitete Analysetag (YYYY-MM-DD).
        start (str): Start-Uhrzeit des Durchlaufs (HH:MM:SS).
        dur_total (float): Gesamtlaufzeit der Pipeline für diesen Tag.
        dur_search (float): Laufzeit des Web-Retrievals.
        dur_filter (float): Laufzeit der Relevanzprüfung/Filterung.
        dur_ki (float): Laufzeit der Textgenerierung (LLM).
        delay (float): Netzweite Verspätungsminuten an diesem Tag.
        sources (int): Anzahl der als relevant eingestuften Quellen.
        status (str): Ergebnisstatus ('Erfolg' oder spezifischer Fehler).
    """
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow([date, start, dur_total, dur_search, dur_filter, dur_ki, round(delay, 2), sources, status])

def write_markdown(date: str, delay: float, res: dict) -> None:
    """Formatiert den generierten Bericht und die Quellen in sauberes Markdown.
    Hängt das Ergebnis im deutschen Datumsformat an das Master-File an.
    
    Args:
        date (str): Das verarbeitete Datum (YYYY-MM-DD).
        delay (float): Die gemessene Durchschnittsverspätung.
        res (dict): Das Resultat-Dictionary aus der ReasoningEngine.
    """
    dt_formatted = datetime.strptime(date, '%Y-%m-%d').strftime('%d.%m.%Y')
    
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(f"## Analyse für den {dt_formatted}\n")
        f.write(f"- **Gemessene Durchschnittsverspätung:** {delay:.2f} Minuten\n")
        f.write(f"- **Generierter Analysebericht:**\n\n{res['answer']}\n\n")
        f.write("### Verwendete Quellen:\n")
        if res.get('sources'):
            for s in res['sources']:
                f.write(f"- [{s['title']}]({s['url']})\n")
        else:
            f.write("- Keine spezifischen Quellen für diesen Tag verifiziert.\n")
        f.write("\n---\n\n")

if __name__ == "__main__":
    run_batch()