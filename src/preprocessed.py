"""Modul zur Vorverarbeitung und Zusammenführung der Schienenverkehrs-Rohdaten.
Dieses Skript liest monatliche Daten im Parquet-Format ein, filtert
die Datensätze auf Züge des Fernverkehrs (ICE, IC, EC) und führt diese
zu einem bereinigten, performanten Master-Datensatz zusammen.
"""

import pandas as pd
import os
import glob
from typing import List

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def main() -> None:
    """Führt die Pipeline für die Bereinigung der Betriebsmeldungen aus."""
    print("Starte Daten-Bereinigung...\n")

    file_paths: List[str] = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    print(f"{len(file_paths)} Dateien gefunden.\n")

    filtered_dfs: List[pd.DataFrame] = []

    for file in file_paths:
        print(f"Verarbeite {os.path.basename(file)}...")
        df = pd.read_parquet(file)
        
        df_filtered = df[df['train_type'].isin(['ICE', 'IC', 'EC'])]
        
        filtered_dfs.append(df_filtered)
        print(f"   -> {len(df_filtered):,} Fernverkehr-Halte gefiltert (von {len(df):,}).")

    print("\nFühre alle Monate zusammen...")
    data_df = pd.concat(filtered_dfs, ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "ice_data.parquet")
    data_df.to_parquet(output_path)

    print(f"Datensatz hat insgesamt {len(data_df):,} Zeilen.")
    print(f"Gespeichert unter: {output_path}")


if __name__ == "__main__":
    main()