import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Daten-Übersicht")
st.title("Daten-Übersicht: Alle Monate")

monate = ["2024-07"]
#monate = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01"]

@st.cache_data
def lade_monat(monat):
    dateipfad = f"data/raw/data-{monat}.parquet"
    return pd.read_parquet(dateipfad).head(30)

for monat in monate:
    st.subheader(f"Monat: {monat}")
    df_ausschnitt = lade_monat(monat)
    st.dataframe(df_ausschnitt, use_container_width=True)