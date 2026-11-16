"""Hauptprogramm und Einstiegspunkt für das Dashboard.
Dieses Skript initialisiert die Streamlit-Benutzeroberfläche, verwaltet die
Session States, bindet die Visualisierungskomponenten ein
und steuert die benutzergetriggerte Ausführung der kausalen RAG-Pipeline.
"""

import streamlit as st
import pandas as pd
import time
import os

from reasoning_engine import ReasoningEngine
from config import SUPPORTED_MODELS
import data_manager as dm
import ui_components as ui

st.set_page_config(layout="wide", page_title="bachelor-thesis-dashboard")
ui.inject_custom_css()

full_df, daily_data, threshold_val, avg_delay_global, std_delay_global, context_data, kpi_data = dm.load_all_data()
daily_data['datum'] = pd.to_datetime(daily_data['datum'])

if 'last_analysis' not in st.session_state:
    st.session_state['last_analysis'] = None
if 'last_analyzed_date' not in st.session_state:
    st.session_state['last_analyzed_date'] = None

with st.sidebar:
    st.header("Konfiguration")
    
    if 'selected_model_id' not in st.session_state:
        default_label = list(SUPPORTED_MODELS.keys())[3]
        st.session_state['selected_model_id'] = SUPPORTED_MODELS[default_label]["id"]
        st.session_state['selected_ctx'] = SUPPORTED_MODELS[default_label]["ctx"]

    model_label = st.selectbox("KI-Modell wählen:", options=list(SUPPORTED_MODELS.keys()), index=3)
    st.session_state['selected_model_id'] = SUPPORTED_MODELS[model_label]["id"]
    st.session_state['selected_ctx'] = SUPPORTED_MODELS[model_label]["ctx"]
    
    st.divider()
    if st.button("System-Cache leeren", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.caption(f"Aktives Modell: \n{st.session_state['selected_model_id']}")

@st.cache_resource
def get_engine(m_name: str) -> ReasoningEngine:
    """Initialisiert und cacht die Instanz der ReasoningEngine für die Pipeline.

    Args:
        m_name (str): Die ID des aktuell ausgewählten primären LLMs.

    Returns:
        ReasoningEngine: Die einsatzbereite Analyse-Engine-Instanz.
    """
    return ReasoningEngine(model_name=m_name, validator_model='llama3.1')

engine = get_engine(st.session_state['selected_model_id'])

st.header("Analyse der Fernverkehr-Zugverspätungen")
st.markdown("Das Dashboard identifiziert Anomalien von IC und ICE Zügen im Netz der Deutschen Bahn für den Zeitraum **Juli 2024 - Januar 2026**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Durchschnittliche Verspätung pro Tag", f"{avg_delay_global:.2f} Min")
k2.metric("Grenzwert / Threshold (μ+2σ)", f"{threshold_val:.2f} Min")
k3.metric("Gefundene Anomalien", f"{daily_data['is_anomaly'].sum()}")
k4.metric("Durchschnittliche Pünktlichkeit", f"{kpi_data['punctuality']:.1f}%")

f1, f2, f3, f4 = st.columns(4)
f1.metric("Anzahl aller Datenpunkte", f"{kpi_data['total_n']:,}") 
f2.metric("Bahnhöfe im Netz", f"{kpi_data['unique_stations']}")
f3.metric("Max. Verspätung", f"{kpi_data['max_delay']} Min")

st.divider()

st.subheader("1. Zeitverlauf & Trend")
col_f1, col_f2, col_f3 = st.columns([1, 1, 1])

with col_f1:
    available_years = sorted(daily_data['datum'].dt.year.unique())
    sel_years = st.multiselect("Jahre filtern:", options=available_years)
with col_f2:
    sel_months = st.multiselect("Monate filtern:", options=list(range(1, 13)), 
                                format_func=lambda x: ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"][x-1])
with col_f3:
    st.write("**Anzeige-Optionen:**")
    c_daily = st.checkbox("Tageswerte anzeigen", value=True)
    c_trend = st.checkbox("7-Tage-Trend anzeigen", value=False)

filtered_df = daily_data.copy()
if sel_years:
    filtered_df = filtered_df[filtered_df['datum'].dt.year.isin(sel_years)]
if sel_months:
    filtered_df = filtered_df[filtered_df['datum'].dt.month.isin(sel_months)]

chart = ui.plot_main_chart(filtered_df, threshold_val, show_daily=c_daily, show_trend=c_trend)
chart = chart.configure_axisX(
    labelExpr="month(datum.value) === 0 ? timeFormat(datum.value, '%Y') : ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember'][month(datum.value)]",
    labelAngle=-45,
    labelFontSize=14,
    titleFontSize=12
).configure_axisY(
    labelFontSize=11,
    titleFontSize=12
).configure_line(
    strokeWidth=2
).configure_point(
    size=50
)
st.altair_chart(chart, width='stretch')

st.divider()

st.subheader("2. Ursachenanalyse")

col_cal, _ = st.columns([1, 3])
with col_cal:
    selected_day = st.date_input("Analysetag wählen:", 
                                 value=daily_data[daily_data['is_anomaly']]['datum'].max(),
                                 min_value=daily_data['datum'].min(),
                                 max_value=daily_data['datum'].max())

day_str = selected_day.strftime("%Y-%m-%d")
day_data = daily_data[daily_data['datum'] == pd.Timestamp(selected_day)]

if not day_data.empty:
    delay = day_data['delay_in_min'].values[0]
    is_anom = day_data['is_anomaly'].values[0]
    
    prev_day = selected_day - pd.Timedelta(days=1)
    prev_day_data = daily_data[daily_data['datum'] == pd.Timestamp(prev_day)]
    yesterday_anom = prev_day_data['is_anomaly'].values[0] if not prev_day_data.empty else False
    
    next_day = selected_day + pd.Timedelta(days=1)
    next_day_data = daily_data[daily_data['datum'] == pd.Timestamp(next_day)]
    tomorrow_anom = next_day_data['is_anomaly'].values[0] if not next_day_data.empty else False

    with st.container(border=True):
        st.markdown(f"### Steckbrief: {selected_day.strftime('%d.%m.%Y')}")
        c1, c2, c3 = st.columns([1, 1, 2.2]) 
        
        with c1:
            st.metric("Durchschnittliche Verspätung", f"{delay:.2f} Min", 
                      delta=f"{delay - avg_delay_global:.2f} Min vs. Durchschnitt", delta_color="inverse")
            if day_str in context_data:
                ctx = context_data[day_str]
                st.metric("Signifikanz", f"{ctx.get('severity_z_score', 'N/A')} σ")
                st.metric("Hotspot-Einfluss", f"{ctx.get('concentration_ratio', 0)}%")
                st.metric(label="Anzahl der Bahnhöfe", value=f"{ctx.get('n_stations', 0)}")

        with c2:
            st.write("**System-Bewertung:**")
            if is_anom: st.error("Anomalie erkannt")
            else: st.success("Normalbetrieb")
            
            st.write("**Zeitlicher Kontext:**")
            if yesterday_anom: st.warning("Gestern: Anomalie")
            else: st.success("Normalbetrieb")
            if tomorrow_anom: st.warning("Morgen: Anomalie")
            else: st.success("Normalbetrieb")
                
            if day_str in context_data:
                with st.expander("Top Hotspots anzeigen", expanded=False):
                    for station in context_data[day_str].get('top_affected_stations', []):
                        st.write(f"- {station}")

        with c3:
            window_df = daily_data[(daily_data['datum'] >= pd.Timestamp(selected_day) - pd.Timedelta(days=3)) & 
                                   (daily_data['datum'] <= pd.Timestamp(selected_day) + pd.Timedelta(days=3))].copy()
            st.altair_chart(ui.plot_mini_preview(window_df, pd.Timestamp(selected_day), threshold_val), width='stretch')

    if st.button("Ursachenanalyse starten", type="primary", width='stretch'):
        status_container = st.empty()
        try:
            full_start = time.time()
            prev_day_str = (selected_day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            next_day_str = (selected_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            def get_clean_ctx(d_str):
                c = context_data.get(d_str, {})
                return {"severity_z_score": c.get('severity_z_score', 0), "event_type": c.get('event_type', 'Normal')}

            temporal_context = {"prev": get_clean_ctx(prev_day_str), "next": get_clean_ctx(next_day_str)}
            
            status_container.info("Schritt 1/3: Web-Recherche via Tavily läuft...")
            start_search = time.time()
            news_results = engine.search_news_for_date(day_str)
            search_dur = round(time.time() - start_search, 2)
            
            status_container.info(
                f"Recherche abgeschlossen ({search_dur}s)\n\n"
                f"Schritte 2 & 3: Relevanzprüfung & KI-Kausalanalyse laufen... (Dies kann 2-3 Minuten dauern)"
            )
            
            result_data = engine.generate_explanation(
                day_str, delay, avg_delay_global, threshold_val, news_results, 
                context_size=st.session_state['selected_ctx'], temporal_context=temporal_context
            )
            
            result_data['duration'] = round(time.time() - full_start, 1)
            
            if 'timings' not in result_data: 
                result_data['timings'] = {'filtering': 0.0, 'final_analysis': 0.0}
            
            result_data['timings']['search'] = search_dur
            
            status_container.success(f"Analyse erfolgreich beendet! (Gesamtdauer: {result_data['duration']}s)")
            time.sleep(1)
            
            st.session_state['last_analysis'] = result_data
            st.session_state['last_analyzed_date'] = day_str
            st.session_state['rerun_trigger'] = True
            st.rerun()
        except Exception as e:
            status_container.empty()
            st.error(f"Fehler in der RAG-Pipeline: {e}")

    if st.session_state.get('last_analyzed_date') == day_str and st.session_state.get('last_analysis'):
        res = st.session_state['last_analysis']
        st.divider()
        
        t = res.get('timings', {})
        p1, p2, p3, p4 = st.columns(4)
        p1.caption(f"Recherche: {t.get('search', 0)}s")
        p2.caption(f"Relevanzprüfung: {t.get('filtering', 0)}s")
        p3.caption(f"KI-Analyse: {t.get('final_analysis', 0)}s")
        p4.caption(f"Gesamtdauer: {res.get('duration', 0)}s")

        col_text, col_sources = st.columns([2, 1])
        with col_text:
            st.subheader("RAG-Ursachenanalyse")
            with st.container(border=True):
                st.markdown(res['answer'])                
                st.divider()
                st.markdown(
                    "**Bewertung der Analyse**", 
                    help=(
                        "1 = sehr schlecht: Analyse widerspricht der Realität komplett.\n\n"
                        "10 = perfekte Analyse: Alle Aussagen durch Quellen belegbar und logische Schlussfolgerung."
                    )
                )
                
                rating = st.select_slider(
                    "Wie hilfreich beurteilst du diese Erklärung?",
                    options=list(range(1, 11)), 
                    value=5, 
                    key=f"rate_widget_{day_str}"
                )

                user_comment = st.text_area(
                    "Anmerkungen (Optional):", 
                    placeholder="Was war gut? Was hat gefehlt?",
                    key=f"comment_widget_{day_str}"
                )
                
                if st.button("Bewertung absenden", width='stretch', key=f"btn_widget_{day_str}"):
                    dm.save_feedback(
                        date_analyzed=day_str,
                        model_name=st.session_state['selected_model_id'],
                        score=rating,
                        delay_value=delay,
                        comment=user_comment 
                    )
                    st.success("Feedback gespeichert!")
         
        with col_sources:
            st.subheader("Belege & Quellen")
            if res.get('sources'):
                for s in res['sources']:
                    st.info(f"**[{s['title']}]({s['url']})**")
            else:
                st.write("Keine spezifischen Quellbelege gefunden.")

st.divider()

st.subheader("3. Zusätzliche Daten und Visualisierungen")

t1, t2, t3, t4, t5 = st.tabs([
    "Kalender-Heatmap", 
    "Regionale Hotspots", 
    "Wochentags-Profil", 
    "Verspätungs-Verteilung", 
    "Anomalie-Störungsprofil"
])

station_stats = dm.get_station_stats_for_day(full_df, selected_day)

with t1:
    st.altair_chart(ui.plot_calendar_heatmap(daily_data), width='stretch')
    st.caption("Grün/Gelb: Normalbetrieb | Dunkelrot: Schwere netzweite Störungen.")

with t2:
    if not station_stats.empty:
        col_c, col_t = st.columns([1.5, 1])
        with col_c: st.altair_chart(ui.plot_impact_scatter(station_stats), width='stretch')
        with col_t: st.dataframe(station_stats, width='stretch', height=500)
        st.caption("Das Blasendiagramm setzt das Verkehrsaufkommen in Relation zur Verspätung. Die Kreisgröße visualisiert den kumulierten operativen Störungs-Impact des Bahnhofs.")
    else: st.warning("Keine Detaildaten für diesen Tag vorhanden.")

with t3:
    st.altair_chart(ui.plot_weekday_boxplot(daily_data), width='stretch')
    st.caption("Boxplots zeigen Median und Ausreißer. Ideal zur Identifikation von Pendler-Hotspots.")

with t4:
    st.altair_chart(ui.plot_delay_histogram(daily_data), width='stretch')
    st.caption("Histogramm der Tagesmittelwerte zur Validierung der Grenzwert-Definition.")

with t5:
    if day_str in context_data:
        ctx = context_data[day_str]
        curr = {'delay': delay, 'z_score': ctx.get('severity_z_score', 0), 
                'impact': ctx.get('concentration_ratio', 0), 'stations': ctx.get('n_stations', 0), 'max': delay * 1.5}
        avg = {'delay': avg_delay_global, 'impact': 15, 'stations': kpi_data['unique_stations']/30, 'max': avg_delay_global*2}
        st.plotly_chart(ui.plot_anomaly_radar(curr, avg), width='stretch')
        st.caption("Das Netzdiagramm vergleicht das aktuelle Tagesprofil (Rot) mit dem historischen Netzdurchschnitt (Grau) zur mehrdimensionalen Charakterisierung der Störung.")
    else: st.info("Wähle einen Anomalie-Tag aus, um das Diagramm zu sehen.")