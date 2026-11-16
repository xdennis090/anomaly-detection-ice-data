import pandas as pd

df = pd.read_parquet("data/processed/ice_master_data.parquet")
anzahl_tage = df['time'].dt.date.nunique()
print(f"Datensatz umfasst {anzahl_tage} Tage.")