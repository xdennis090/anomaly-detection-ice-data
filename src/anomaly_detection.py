"""Modul zur statistischen Identifikation und Profilierung von Anomalien.
Dieses Skript berechnet den dynamischen Anomalie-Schwellenwert auf Basis der
Standardabweichung, isoliert signifikante Ausreißertage und extrahiert
fallspezifische Wissensmerkmale wie die geografische Konzentrationsrate.
"""

import pandas as pd
import os
import json

INPUT_PATH = os.path.join("data", "processed", "ice_data.parquet")
OUTPUT_CSV = os.path.join("data", "processed", "detected_anomalies.csv")
OUTPUT_JSON = os.path.join("data", "processed", "anomaly_context.json")

def create_anomaly_profile() -> None:
    """Identifiziert statistische Anomalien und erstellt detaillierte Kontextprofile.
    Die Funktion aggregiert die täglichen Verspätungswerte, ermittelt über
    den Grenzwert (µ + 2σ) alle anomalen Betriebstage und berechnet für diese
    den Schweregrad (Z-Score) sowie die räumliche Konzentrationsrate.
    Die Ergebnisse werden als CSV-Matrix und strukturierte JSON-Kontextbasis persistiert.
    """
    if not os.path.exists(INPUT_PATH):
        print(f"Abbruch: Eingabedatei {INPUT_PATH} not gefunden.")
        return

    print("Lade und berechne Daten...")
    df = pd.read_parquet(INPUT_PATH)
    df['datum'] = pd.to_datetime(df['time']).dt.date

    daily_stats = df.groupby('datum')['delay_in_min'].mean().reset_index()
    global_mean = daily_stats['delay_in_min'].mean()
    global_std = daily_stats['delay_in_min'].std()
    anomaly_threshold = global_mean + (2 * global_std)
    
    daily_stats['is_anomaly'] = daily_stats['delay_in_min'] > anomaly_threshold
    anomalien_df = daily_stats[daily_stats['is_anomaly']].copy()
    anomaly_details = {}

    print(f"Analysiere {len(anomalien_df)} identifizierte Anomalien...")

    for _, row in anomalien_df.iterrows():
        current_date = row['datum']
        day_data = df[df['datum'] == current_date]
        
        station_stats = day_data.groupby('station_name')['delay_in_min'].agg(['mean', 'sum', 'count'])
        valid_stations = station_stats[station_stats['count'] > 5]
        top_stations_df = valid_stations.sort_values(by='mean', ascending=False).head(3)
        
        total_delay_sum = day_data['delay_in_min'].sum()
        top_3_sum = top_stations_df['sum'].sum()
        concentration_ratio = (top_3_sum / total_delay_sum * 100) if total_delay_sum > 0 else 0

        anomaly_details[str(current_date)] = {
            "date": str(current_date),
            "avg_delay": round(row['delay_in_min'], 2),
            "severity_z_score": round((row['delay_in_min'] - global_mean) / global_std, 2),
            "concentration_ratio": round(concentration_ratio, 1),
            "n_stations": int(day_data['station_name'].nunique()),
            "top_affected_stations": top_stations_df.index.tolist(),
            "event_type": "Anomalie erkannt" 
        }

    anomalien_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(anomaly_details, f, ensure_ascii=False, indent=4)
        
    print(f"Erfolg: {len(anomalien_df)} Anomalien in CSV und JSON exportiert.")

if __name__ == "__main__":
    create_anomaly_profile()