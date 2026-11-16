"""Modul für das Datenmanagement, statistische Berechnungen und Feedback.
Dieses Skript kapselt alle Datenzugriffe des Prototyps. Es berechnet globale 
statistische Kennzahlen (Mittelwert, Standardabweichung, Rolling Average), 
generiert tagesbasierte Bahnhofsstatistiken und sichert Benutzer-Feedback in CSV-Strukturen.
"""

import pandas as pd
import streamlit as st
import json
import os
import csv
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

FEEDBACK_FILE = "evaluation/feedback_data.csv"
INPUT_PARQUET = "data/processed/ice_data.parquet"
CONTEXT_JSON = "data/processed/anomaly_context.json"

@st.cache_data
def load_all_data() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, float, float, float, Dict[str, Any], Dict[str, Any]]]:
    """Lädt die gesamte Datenbasis und berechnet die globalen Statistikmetriken.
    Die Funktion ist über Streamlit gecached, um rechenintensive Berechnungen 
    auf den Zeitreihendaten bei wiederholten Aufrufen zu minimieren.

    Returns:
        Optional[Tuple[...]]: Ein Tuple bestehend aus:
            - df (pd.DataFrame): Die bereinigten Rohdaten aller Halte.
            - daily_stats (pd.DataFrame): Aggregierte Verspätungen pro Tag.
            - threshold (float): Der statistische Anomalie-Grenzwert (µ + 2σ).
            - avg_delay_global (float): Der globale arithmetische Mittelwert.
            - std_delay_global (float): Die globale Standardabweichung.
            - context (Dict[str, Any]): Geladene JSON-Kontextdaten für Anomalien.
            - kpi_data (Dict[str, Any]): Globale Stammdaten-KPIs für das Dashboard.
        Falls die Parquet-Datei fehlt, wird None zurückgegeben.
    """
    if not os.path.exists(INPUT_PARQUET):
        st.error(f"Datei nicht gefunden: {INPUT_PARQUET}")
        return None

    df = pd.read_parquet(INPUT_PARQUET)
    df['datum'] = pd.to_datetime(df['time']).dt.normalize()
    
    daily_stats = df.groupby('datum')['delay_in_min'].mean().reset_index()
    daily_stats['rolling_avg'] = daily_stats['delay_in_min'].rolling(window=7, min_periods=1).mean()
    avg_delay_global = daily_stats['delay_in_min'].mean()
    std_delay_global = daily_stats['delay_in_min'].std()
    threshold = avg_delay_global + (2 * std_delay_global)
    
    daily_stats['is_anomaly'] = daily_stats['delay_in_min'] > threshold
    total_n = len(df)
    unique_stations = df['station_name'].nunique()
    max_delay = int(df['delay_in_min'].max())
    punctual_count = (df['delay_in_min'] < 6).sum()
    punctuality_rate = (punctual_count / total_n) * 100
    
    kpi_data = {
        "total_n": total_n,
        "unique_stations": unique_stations,
        "max_delay": max_delay,
        "punctuality": punctuality_rate
    }

    context = {}
    if os.path.exists(CONTEXT_JSON):
        with open(CONTEXT_JSON, "r", encoding="utf-8") as f:
            context = json.load(f)
            
    return df, daily_stats, threshold, avg_delay_global, std_delay_global, context, kpi_data


def get_station_stats_for_day(df: pd.DataFrame, selected_day: Any) -> pd.DataFrame:
    """Fasst die Verspätungswerte einzelner Bahnhöfe für ein spezifisches Datum zusammen.
    Filtert Ausreißer-Stationen mit weniger als 5 Zügen pro Tag heraus, um die 
    Konzentrationsrate sauber abzubilden.

    Args:
        df (pd.DataFrame): Das Master-Dataframe der Fernverkehrshalte.
        selected_day (Any): Das ausgewählte Analysedatum (String oder Date-Objekt).

    Returns:
        pd.DataFrame: Sortierte und bereinigte Bahnhofs-Statistiken des Tages.
    """
    target_date = pd.to_datetime(selected_day).normalize()
    day_data = df[df['datum'] == target_date]
    if day_data.empty:
        return pd.DataFrame()
    
    stats = day_data.groupby('station_name')['delay_in_min'].agg(['mean', 'count']).reset_index()
    stats = stats[stats['count'] > 5].sort_values(by='mean', ascending=False)
    
    return stats.rename(columns={'mean': 'Durchschnittliche Verspätung', 'count': 'Anzahl Züge'})


def save_feedback(date_analyzed: str, model_name: str, score: Any, 
                  delay_value: float, comment: str = "") -> None:
    """Dokumentiert die qualitative und quantitative Bewertung eines Berichts in einer CSV-Datei.
    Erzeugt beim ersten Aufruf automatisch die Tabellen-Header, falls die
    Zieldatei noch nicht im Dateisystem existiert.

    Args:
        date_analyzed (str): Das analysierte Anomalie-Datum (YYYY-MM-DD).
        model_name (str): ID des genutzten LLMs (z.B. 'command-r').
        score (Any): Vergebenes Rating durch den Anwender.
        delay_value (float): Durchschnittsverspätung des Tages.
        comment (str): Optionaler Freitext-Kommentar des Nutzers.
    """
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'analyzed_date', 'model', 'score', 'delay_min', 'comment'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_analyzed,
            model_name,
            score,
            delay_value,
            comment.replace("\n", " ")
        ])