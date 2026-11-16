"""Modul für interaktive UI-Komponenten und datenanalytische Visualisierungen.
Dieses Skript hat sämtliche Grafik- und Chart-Generierungen des Dashboards.
Es nutzt Altair für deklarative, interaktive statistische Diagramme und Plotly 
für fortgeschrittene, multivariate Radar-Analysen von Netz-Anomalien.
"""

import streamlit as st
import altair as alt
import pandas as pd
import os
import plotly.graph_objects as go
from typing import Any, Dict, List


def inject_custom_css() -> None:
    """Lädt das externe CSS-Stylesheet und gibt es in die Streamlit-Instanz.
    Dient der Umsetzung des Benutzeroberflächen-Designs.
    Gibt bei Fehlen der Datei eine kontrollierte Warnung in der UI aus.
    """
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Design-Datei (style.css) nicht gefunden.")


def plot_main_chart(data: pd.DataFrame, anomaly_threshold: float, 
                    show_daily: bool = True, show_trend: bool = True) -> alt.Chart:
    """Generiert das interaktive Haupt-Zeitreihendiagramm der Netzverspätungen.
    Schichtet Rohwerte, den 7-Tage-Trend sowie farblich hervorgehobene Anomalie-Punkte
    übereinander und fügt eine dynamische Grenzwertlinie hinzu.

    Args:
        data (pd.DataFrame): Zeitreihendaten der täglichen Durchschnittsverspätungen.
        anomaly_threshold (float): Der statistische Anomalie-Schwellenwert (µ + 2σ).
        show_daily (bool): Flag zur Anzeige der täglichen Rohwerte-Kurve.
        show_trend (bool): Flag zur Anzeige der geglätteten Trend-Kurve.

    Returns:
        alt.Chart: Ein geschichtetes, interaktives Altair-Diagramm-Objekt.
    """
    layers = []
    hover_selection = alt.selection_point(on='mouseover', nearest=True, fields=['datum'], empty=False)

    base = alt.Chart(data).encode(
        x=alt.X('datum:T', title='Zeitraum (Tage)'),
        tooltip=[
            alt.Tooltip('datum:T', title='Datum'), 
            alt.Tooltip('delay_in_min:Q', title='Durchschnittliche Verspätung (Min)', format='.2f'),
            alt.Tooltip('rolling_avg:Q', title='7-Tage-Trend (Min)', format='.2f')
        ]
    )

    if show_daily:
        line_daily = base.mark_line(color="#3284ff", strokeWidth=1.5, opacity=0.4).encode(
            y=alt.Y('delay_in_min:Q', title='Durchschnittliche Verspätung (Min)')
        )
        layers.append(line_daily)

    if show_trend:
        line_trend = base.mark_line(color="#19ff4b", strokeWidth=3).encode(
            y=alt.Y('rolling_avg:Q')
        )
        layers.append(line_trend)

    anomaly_points = base.mark_point(filled=True, size=80).encode(
        y='delay_in_min:Q',
        color=alt.condition(alt.datum.is_anomaly, alt.value('#e74c3c'), alt.value('transparent')),
        opacity=alt.condition(alt.datum.is_anomaly, alt.value(1), alt.value(0)),
        size=alt.condition(hover_selection, alt.value(150), alt.value(80))
    ).add_params(hover_selection).transform_filter(alt.datum.is_anomaly == True)
    
    layers.append(anomaly_points)

    threshold_line = alt.Chart(pd.DataFrame({'y': [anomaly_threshold]})).mark_rule(
        color='#f39c12', strokeDash=[5,5], strokeWidth=2
    ).encode(y='y')
    
    layers.append(threshold_line)

    return alt.layer(*layers).properties(height=350).interactive()


def plot_mini_preview(window_data: pd.DataFrame, selected_day: Any, 
                      anomaly_threshold: float) -> alt.Chart:
    """Erstellt ein fokussiertes Balkendiagramm für das zeitliche Umfeld einer Anomalie.
    Färbt den spezifischen Untersuchungstag rot ein und hebt ihn von den 
    umgebenden Tagen (Vortag und Folgetag) visuell ab (Drei-Tage-Fenster).

    Args:
        window_data (pd.DataFrame): Gefilterte Datenpunkte des zeitlichen Fensters.
        selected_day (Any): Das aktuell ausgewählte Analysedatum (Fokus-Tag).
        anomaly_threshold (float): Der statistische Anomalie-Schwellenwert.

    Returns:
        alt.Chart: Balkendiagramm mit statischer Schwellenwertlinie.
    """
    df = window_data.drop_duplicates(subset=['datum']).copy()
    df['display_date'] = df['datum'].apply(lambda x: x.strftime('%d.%m.'))
    df['status_color'] = df['datum'].apply(lambda x: 'Fokus' if x == selected_day else 'Umfeld')
    
    bars = alt.Chart(df).mark_bar(size=35).encode(
        x=alt.X('display_date:O', 
                title=None, 
                axis=alt.Axis(labelAngle=0, labelColor='#e6e6e6', titleColor='#e6e6e6'),
                sort=alt.EncodingSortField(field="datum", order="ascending")),
        y=alt.Y('delay_in_min:Q', title='Verspätung (Min)', axis=alt.Axis(labelColor='#e6e6e6', titleColor='#e6e6e6')),
        color=alt.Color('status_color:N', 
                        scale=alt.Scale(domain=['Fokus', 'Umfeld'], range=['#e74c3c', '#34495e']),
                        legend=None),
        tooltip=[
            alt.Tooltip('datum:T', title='Datum'),
            alt.Tooltip('delay_in_min:Q', title='Verspätung (Min)', format='.2f')
        ]
    )
    
    threshold_rule = alt.Chart(pd.DataFrame({'y': [anomaly_threshold]})).mark_rule(
        color='#f39c12', strokeDash=[4,4], strokeWidth=2
    ).encode(y='y')
    
    return (bars + threshold_rule).configure_view(strokeOpacity=0).properties(height=300).configure_axis(labelColor='#e6e6e6', titleColor='#e6e6e6')


def plot_impact_bar_chart(impact_data: pd.DataFrame) -> alt.Chart:
    """Generiert ein horizontales Balkendiagramm der Top 20 verspäteten Bahnhöfe.
    Bildet die räumliche Komponente einer Störung ab, sortiert nach absteigender Intensität.

    Args:
        impact_data (pd.DataFrame): Aggregierte Bahnhofs-Statistiken des Analysetages.

    Returns:
        alt.Chart: Horizontales Balkendiagramm für die Hotspot-Erkennung.
    """
    return alt.Chart(impact_data.head(20)).mark_bar(
        color='#e74c3c', cornerRadiusEnd=4
    ).encode(
        x=alt.X('Durchschnittliche Verspätung:Q', title='Durchschnittliche Verspätung (Min)'),
        y=alt.Y('station_name:N', sort='-x', title=None),
        tooltip=[
            alt.Tooltip('station_name:N', title='Bahnhof'),
            alt.Tooltip('Durchschnittliche Verspätung:Q', title='Durchschnittliche Verspätung (Min)', format='.2f'),
            alt.Tooltip('Anzahl Züge:Q', title='Gezählte Züge')
        ]
    ).properties(height=500)


def plot_seasonal_overlay(df: pd.DataFrame, threshold: float) -> alt.Chart:
    """Erstellt ein saisonales Overlay zur Visualisierung von Jahresverläufen.
    Projiziert reale Datenpunkte unterschiedlicher Kalenderjahre auf eine 
    virtuelle Zeitachse, um wiederkehrende Muster aufzuzeigen.

    Args:
        df (pd.DataFrame): Tägliche Verspätungsstatistiken über alle Jahre.
        threshold (float): Globaler Anomalie-Grenzwert.

    Returns:
        alt.Chart: Interaktives Liniendiagramm mit Legenden-Filterung.
    """
    plot_df = df.copy()
    plot_df['datum'] = pd.to_datetime(plot_df['datum'])
    plot_df['Jahr'] = plot_df['datum'].dt.year.astype(str)
    plot_df['virt_datum'] = plot_df['datum'].apply(lambda x: x.replace(year=2000))
    domain = ['2024', '2025', '2026']
    range_ = ['#4169E1', '#FFA500', '#FF4B4B']
    
    selection = alt.selection_point(fields=['Jahr'], bind='legend')
    
    lines = alt.Chart(plot_df).mark_line(
        interpolate='monotone', 
        strokeWidth=2
    ).encode(
        x=alt.X('virt_datum:T', 
                scale=alt.Scale(domain=['2000-01-01', '2000-12-31']),
                axis=alt.Axis(format='%b', title='Jahresverlauf', labelAngle=0, tickCount='month'),
                title='Monat (Jahresvergleich)'),
        y=alt.Y('delay_in_min:Q', title='Durchschnittliche Verspätung (Min)'),
        color=alt.Color('Jahr:N', 
                        scale=alt.Scale(domain=domain, range=range_), 
                        title='Kalenderjahr'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[
            alt.Tooltip('datum:T', title='Echtes Datum'),
            alt.Tooltip('delay_in_min:Q', title='Verspätung', format='.2f')
        ]
    )

    threshold_rule = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(
        color='#FF4B4B',
        strokeDash=[5, 5],
        size=1.5
    ).encode(y='y:Q')

    return (lines + threshold_rule).add_params(selection).properties(height=400).interactive()


def plot_calendar_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Generiert eine Kalender-Heatmap nach Wochentagen und Kalenderwochen.
    Erlaubt eine intuitive, farbcodierte Erfassung langfristiger Störungscluster 
    über Ampelfarben direkt in der Dashboard-Hauptanzeige.

    Args:
        df (pd.DataFrame): Gesamte tagesbasierte Verspätungszeitreihe.

    Returns:
        alt.Chart: Gefacetter, großflächiger Kachel-Katalog.
    """
    plot_df = df.copy()
    plot_df['datum'] = pd.to_datetime(plot_df['datum'])
    plot_df['Jahr'] = plot_df['datum'].dt.year
    plot_df['Woche'] = plot_df['datum'].dt.isocalendar().week
    plot_df['Wochentag'] = plot_df['datum'].dt.day_of_week 
    plot_df['Tag_Name'] = plot_df['datum'].dt.day_name()
    plot_df['Monat'] = plot_df['datum'].dt.month_name()

    wochentage = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    color_scale = alt.Scale(
        domain=[0, 10, 18, 30], 
        range=['#228B22', '#FFD700', '#FF4B4B', '#8B0000'] 
    )
    chart = alt.Chart(plot_df).mark_rect(stroke='black', strokeWidth=0.5).encode(
        x=alt.X('Woche:O', title='Kalenderwoche', axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('Tag_Name:O', 
                sort=wochentage, 
                title=None,
                axis=alt.Axis(labelFontSize=12)),
        color=alt.Color('delay_in_min:Q', 
                        scale=color_scale, 
                        legend=alt.Legend(title="Durchschnittliche Verspätung (Min)"),
                        title='Verspätung'),
        tooltip=[
            alt.Tooltip('datum:T', title='Datum'),
            alt.Tooltip('delay_in_min:Q', title='Verspätung', format='.2f'),
            alt.Tooltip('is_anomaly:N', title='Anomalie')
        ]
    ).properties(
        width=1200, 
        height=150
    ).facet(
        row=alt.Row('Jahr:O', title=None, header=alt.Header(labelFontSize=16, labelAngle=0)),
        spacing=20
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    )

    return chart


def plot_impact_scatter(stats: pd.DataFrame) -> alt.Chart:
    """Erstellt ein interaktives Scatter Plot für Bahnhofs-Metriken.
    Setzt das Verkehrsaufkommen in Relation zur Verspätung und visualisiert 
    den resultierenden geografischen Störungs-Impact über die Punktgröße.

    Args:
        stats (pd.DataFrame): Bahnhofsstatistiken des ausgewählten Tages.

    Returns:
        alt.Chart: Interaktives, zoombares Blasen-Diagramm.
    """
    stats['Impact'] = stats['Durchschnittliche Verspätung'] * stats['Anzahl Züge']
    stats['Impact'] = stats['Impact'].round(2)
    stats['Durchschnittliche Verspätung'] = stats['Durchschnittliche Verspätung'].round(2)
    
    chart = alt.Chart(stats).mark_circle(opacity=0.7, stroke='#0e1117', strokeWidth=0.5).encode(
        x=alt.X('Anzahl Züge:Q', title='Verkehrsaufkommen (Züge)'),
        y=alt.Y('Durchschnittliche Verspätung:Q', title='Durchschnittliche Verspätung (Min)'),
        size=alt.Size('Impact:Q', scale=alt.Scale(range=[100, 1500]), legend=None),
        color=alt.Color('Durchschnittliche Verspätung:Q', scale=alt.Scale(scheme='yelloworangered'), legend=None),
        tooltip=[
            alt.Tooltip('station_name:N', title='Bahnhof'),
            alt.Tooltip('Durchschnittliche Verspätung:Q', title='Verspätung', format='.2f'),
            alt.Tooltip('Anzahl Züge:Q', title='Züge'),
            alt.Tooltip('Impact:Q', title='Impact-Score', format='.2f')
        ]
    ).properties(
        height=350
    ).interactive()
    
    chart = chart.configure_axis(
        labelFontSize=11,
        titleFontSize=12,
        labelColor='#e6e6e6',
        titleColor='#e6e6e6',
        domain=True,
        domainWidth=1.5,
        domainColor='#e6e6e6',
        tickColor='#e6e6e6'
    ).configure_view(
        strokeOpacity=0
    )
    
    return chart


def plot_weekday_boxplot(df: pd.DataFrame) -> alt.Chart:
    """Generiert Boxplots zur Analyse der Verspätungsvarianz nach Wochentagen.
    Erlaubt die Identifizierung wochentagsspezifischer Belastungsmuster im Netz.

    Args:
        df (pd.DataFrame): Tägliche Verspätungszeitreihe.

    Returns:
        alt.Chart: Konfiguriertes Box-Whisker-Diagramm.
    """
    df['Wochentag'] = df['datum'].dt.day_name()
    order = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    
    chart = alt.Chart(df).mark_boxplot(extent='min-max', color='#ff4b4b').encode(
        x=alt.X('Wochentag:N', sort=order, title="Wochentag", axis=alt.Axis(
                    labelAngle=-45,
                    labelColor='#e6e6e6',
                    titleColor='#e6e6e6'
                )),
        y=alt.Y('delay_in_min:Q', title="Verspätung (Min)"),
        tooltip=[
            alt.Tooltip('Wochentag:N'),
            alt.Tooltip('delay_in_min:Q', title='Verspätung', format='.2f')
        ]
    ).properties(height=450, title="Varianz nach Wochentagen")
    
    return chart.configure(numberFormat='.2f')


def plot_delay_histogram(df: pd.DataFrame) -> alt.Chart:
    """Erstellt ein Häufigkeits-Histogramm der netzweiten Verspätungswerte.
    Visualisiert die statistische Verteilung der berechneten Tagesmittelwerte in Klassen.

    Args:
        df (pd.DataFrame): Tägliche Verspätungszeitreihe.

    Returns:
        alt.Chart: Klassifiziertes Säulendiagramm.
    """
    chart = alt.Chart(df).mark_bar(color='#ffa424').encode(
        x=alt.X("delay_in_min:Q", bin=alt.Bin(maxbins=30), title="Verspätungsklasse (Min)"),
        y=alt.Y('count()', title="Anzahl der Tage"),
        tooltip=[
            alt.Tooltip('delay_in_min:Q', bin=alt.Bin(maxbins=30), title='Bereich', format='.2f'),
            alt.Tooltip('count()', title='Anzahl Tage')
        ]
    ).properties(height=450, title="Häufigkeitsverteilung der Verspätungswerte")
    
    return chart.configure(numberFormat='.2f')


def plot_anomaly_radar(day_stats: Dict[str, Any], avg_stats: Dict[str, Any]) -> go.Figure:
    """Erstellt ein Radar Chart zur Anomalie-Charakterisierung.
    Vergleicht fünf Kernmetriken des aktuellen Untersuchungstages direkt mit dem 
    historischen Systemdurchschnitt, um den Profil-Ausprägungstyp sofort sichtbar zu machen.

    Args:
        day_stats (Dict[str, Any]): Metriken des spezifischen Analosetages.
        avg_stats (Dict[str, Any]): Historische Durchschnittswerte des Systems.

    Returns:
        go.Figure: Ein konfiguriertes interaktives Plotly-Polar-Diagrammobjekt.
    """
    categories = ['Durchschnittliche Verspätung', 'Schwere (Z-Score)', 'Hotspot-Einfluss', 'Betroffene Bahnhöfe', 'Maximale Verspätung']
    
    real_values_day = [
        day_stats['delay'], 
        day_stats['z_score'], 
        day_stats['impact'], 
        day_stats['stations'], 
        day_stats['max']
    ]
    
    real_values_avg = [
        avg_stats['delay'], 
        1.0,
        avg_stats['impact'], 
        avg_stats['stations'], 
        avg_stats['max']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[day_stats['delay'], day_stats['z_score']*5, day_stats['impact'], day_stats['stations']/10, day_stats['max']/10],
        theta=categories,
        fill='toself',
        name='Analysetag',
        line_color='#ff4b4b',
        fillcolor='rgba(255, 75, 75, 0.15)',
        customdata=real_values_day,
        hovertemplate="<b>%{theta}</b><br>Wert: %{customdata:.2f} <extra></extra>"
    ))

    fig.add_trace(go.Scatterpolar(
        r=[avg_stats['delay'], 5, avg_stats['impact'], avg_stats['stations']/10, avg_stats['max']/10],
        theta=categories,
        fill='toself',
        name='Durchschnitt',
        line_color='#7f8c8d',
        fillcolor='rgba(127, 140, 141, 0.05)',
        customdata=real_values_avg,
        hovertemplate="<b>%{theta}</b><br>Durchschnitt: %{customdata:.2f} <extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",         
        width=750,   
        height=650,  
        
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, 
                gridcolor="#2d3436",
                linecolor="#e6e6e6",
                tickfont=dict(size=14, color="#e6e6e6"), 
                showticklabels=True 
            ),
            angularaxis=dict(
                gridcolor="#2d3436",      
                tickfont=dict(size=16, color="#e6e6e6", family="Arial"), 
            )
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", 
        
        font=dict(
            family="Arial",
            size=14,
            color="#e6e6e6"
        ),
        legend=dict(
            font=dict(size=15, color="#e6e6e6"),
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=110, r=110, t=70, b=50), 
    )
    
    return fig