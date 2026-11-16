"""Engine für das automatisierte Web-Retrieval und die Kausalanalyse.
Dieses Modul implementiert die ReasoningEngine. Sie steuert die asynchrone 
Websuche, bereinigt Treffer über zeitliche Heuristiken und einen agentischen 
LLM-Filter und generiert strukturierte Ursachenanalysen (RCA).
"""

import os
import re
import json
import ollama
import time
from dotenv import load_dotenv
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ReasoningEngine:
    """Klasse zur Durchführung des Web-Retrievals und der KI-gestützten Analyse."""

    def __init__(self, model_name: str = 'command-r', validator_model: str = 'llama3.1') -> None:
        """Initialisiert die Instanz der ReasoningEngine.

        Args:
            model_name (str): ID des primären Sprachmodells für die Analyse.
            validator_model (str): ID des Modells für die Relevanzprüfung.

        Raises:
            ValueError: Wenn der erforderliche Tavily-API-Schlüssel fehlt.
        """
        if not TAVILY_API_KEY:
            raise ValueError("Konnte TAVILY_API_KEY in der .env Datei nicht finden.")
        
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.model_name = model_name
        self.validator_model = validator_model

    def _format_date(self, date_str: str) -> str:
        """Konvertiert ein ISO-Datum in das deutsche Standardformat.

        Args:
            date_str (str): Das Ausgangsdatum (YYYY-MM-DD).

        Returns:
            str: Das formatierte Datum (DD.MM.YYYY).
        """
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%d.%m.%Y')

    def search_news_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Führt eine parallele Websuche über vier thematische Suchprofile aus.

        Args:
            date_str (str): Das Zieldatum des Vorfalls (YYYY-MM-DD).

        Returns:
            List[Dict[str, Any]]: Eine Liste eindeutiger Suchergebnisse (URLs).
        """
        formatted_date = self._format_date(date_str)
        
        queries = [
            f'Deutsche Bahn Fernverkehr ICE IC Störung Ursache {formatted_date}',
            f'Zugverspätungen Großstörung Bahn News {formatted_date}',
            f'DB Streckensperrung Hauptachse Chaos {formatted_date}',
            f'Bahnstrecke gesperrt Fernverkehr {formatted_date}'
        ]
        
        all_results = []

        def fetch_tavily(query: str) -> List[Dict[str, Any]]:
            try:
                response = self.tavily.search(query=query, search_depth="advanced", max_results=10)
                return response.get('results', [])
            except Exception as e:
                print(f"Tavily Fehler bei '{query}': {e}")
                return []

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            results_lists = list(executor.map(fetch_tavily, queries))

        for res_list in results_lists:
            all_results.extend(res_list)
            
        unique_results = {res['url']: res for res in all_results}.values()
        return list(unique_results)

    def _smart_date_check(self, date_str: str, content: str) -> bool:
        """Prüft, ob ein Textdokument dem Zielzeitraum entspricht.

        Verhindert die fehlerhafte Auswertung von Berichten aus falschen Jahren.

        Args:
            date_str (str): Das Ziel-Anomaliedatum (YYYY-MM-DD).
            content (str): Der unstrukturierte Textinhalt der Quelle.

        Returns:
            bool: True, wenn das Dokument zeitliche Validitätskriterien erfüllt.
        """
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        year_target = date_str.split('-')[0]
        monate = ["Januar", "Februar", "März", "April", "Mai", "Juni", 
                  "Juli", "August", "September", "Oktober", "November", "Dezember"]
        
        patterns = [
            dt.strftime('%d.%m.%Y'),
            f"{dt.day}.{dt.month}.{dt.year}",
            dt.strftime('%d.%m.'),
            f"{dt.day}. {monate[dt.month-1]}"
        ]
        
        content_lower = content.lower()
        if year_target not in content_lower:
            return False
        return any(p.lower() in content_lower for p in patterns)

    def _filter_relevant_sources(self, date_str: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtert extrahierte Quellen zweistufig nach Relevanz und Gültigkeit.

        Nutzt zuerst heuristische Zeitprüfungen und anschließend eine 
        Klassifizierung durch ein lokales LLM, um Rauschen zu minimieren.

        Args:
            date_str (str): Das Zieldatum der Analyse (YYYY-MM-DD).
            results (List[Dict[str, Any]]): Die ungefilterte Liste von Web-Treffern.

        Returns:
            List[Dict[str, Any]]: Die verifizierten, relevanten Quellenobjekte.
        """
        year_target = date_str.split('-')[0]
        if not results:
            return []

        candidates = [res for res in results if self._smart_date_check(date_str, res['content'])]
        
        if len(candidates) == 0:
            return []
        
        if len(candidates) <= 2:
            return candidates

        formatted_date = self._format_date(date_str)
        candidates_text = ""
        for i, res in enumerate(candidates[:15]):
            snippet = res['content'][:350].replace('\n', ' ')
            candidates_text += f"ID: {i} | Titel: {res['title']} | Vorschau: {snippet}...\n"

        validation_prompt = f"""
        Du bist ein präziser Datenanalyst für den Bahnbetrieb. Prüfe die recherchierten Textauszüge für den {formatted_date}. Zieljahr ist strikt: {year_target}
        
        RELEVANZ-KRITERIUM: 
        - Suche nach konkreten Ursachen für überregionale und systemweite Verspätungen im Fernverkehr (ICE/IC/EC). 
        - Typische Beispiele hierfür sind unter anderem Unwetter, großflächige IT- oder Stellwerksausfälle, Oberleitungsschäden auf Hauptachsen, Streiks sowie Notarzteinsätze oder Bombenentschärfungen. 
        - Das Kriterium umfasst jedoch auch ausdrücklich alle anderen Ereignisse, die eine ähnliche netzweite Störwirkung auf den Bahnverkehr entfalten.

        STRIKTER AUSSCHLUSS:
        - Ignoriere kommunale Baustellen, die nur Fußgänger, Gehwege oder den lokalen Straßenverkehr betreffen (z.B. Pflasterarbeiten in Gemeinden).
        - Ignoriere isolierte Ausfälle einzelner Regionalbahnen (RB/RE), es sei denn, sie blockieren eine Hauptfernstrecke.
        - Ignoriere Artikel, die primär über Ereignisse aus anderen Jahren berichten.

        VORLIEGENDE TEXTAUSZÜGE:
        {candidates_text}

        ANTWORT: Gib ausschließlich die IDs der relevanten Quellen als kommagetrennte Liste zurück (z.B. 0, 2). 
        Falls kein Bericht das Kriterium erfüllt, antworte mit 'KEINE'. Schreib keinen Fließtext.
        """
        
        try:
            response = ollama.chat(
                model=self.validator_model, 
                messages=[{'role': 'user', 'content': validation_prompt}],
                options={'temperature': 0.0, 'num_ctx': 8192} 
            )
            reply = response['message']['content'].strip()
            
            if "KEINE" in reply.upper(): 
                return []
                
            selected_ids = [int(s) for s in re.findall(r'\d+', reply)]
            return [candidates[i] for i in selected_ids if i < len(candidates)]
        except Exception as e:
            print(f"Fehler bei Relevanzprüfung: {e}")
            return candidates[:5] 

    def generate_explanation(self, date_str: str, delay_value: float, avg_delay_global: float, 
                             threshold_val: float, news_results: List[Dict[str, Any]], 
                             context_size: int = 8192, temporal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generiert die finale Kausalanalyse auf Basis statistischer und textlicher Daten.
        Baut den finalen XML-basierten RAG-Prompt, bindet den zeitlichen Netzkontext
        ein und fängt leere Wissensbasen über eine strukturierte Nullantwort ab.

        Args:
            date_str (str): Das Untersuchungsdatum (YYYY-MM-DD).
            delay_value (float): Die gemessene Netz-Durchschnittsverspätung.
            avg_delay_global (float): Der historische Erwartungswert der Verspätung.
            threshold_val (float): Der berechnete statistische Anomalie-Grenzwert.
            news_results (List[Dict[str, Any]]): Die Liste der rohen Suchergebnisse.
            context_size (int): Die maximale Größe des LLM-Kontextfensters.
            temporal_context (Optional[Dict[str, Any]]): Zustand von Vortag und Folgetag.

        Returns:
            Dict[str, Any]: Analysebericht, verwendete Quellen und Zeitmessungen.
        """
        year_target = date_str.split('-')[0]
        start_filtering = time.time()
        validated_sources = self._filter_relevant_sources(date_str, news_results)
        filtering_dur = round(time.time() - start_filtering, 2)

        if not validated_sources:
            fallback_answer = f"""### 1. Ursache
            - Keine spezifischen Großereignisse für den Tag {self._format_date(date_str)} gefunden (KUMULATIVE EINFLÜSSE)

            ### 2. Analyse
            - Für den {self._format_date(date_str)} konnten in den verifizierten Nachrichtenquellen keine überregionalen Störungsereignisse für das Zieljahr {year_target} identifiziert werden. 
            - Die gemessene Durchschnittsverspätung von {delay_value:.2f} Minuten basiert somit auf einem erhöhten systemischen Grundrauschen (viele verstreute, nicht-berichtete Mikrostörungen im Schienennetz).

            ### 3. Plausibilitätsprüfung
            - Die statistische Abweichung vom globaler Erwartungswert ({avg_delay_global:.2f} Minuten) lässt sich nicht auf ein singuläres, externes Großereignis zurückführen. 
            - Das Netz lief am {self._format_date(date_str)} ohne gravierende technische Defekte, aber an seiner Kapazitätsgrenze."""
            
            return {
                "answer": fallback_answer,
                "sources": [],
                "timings": {"filtering": filtering_dur, "final_analysis": 0.0}
            }

        lage_bericht = "Die Betriebslage am Vortag war statistisch unauffällig."
        if temporal_context:
            z_prev = temporal_context.get('prev', {}).get('severity_z_score', 0)
            if z_prev > 2.0:
                lage_bericht = "Bereits am Vortag lag eine statistische Anomalie vor, was auf eine persistente Störung hindeutet."
            
            z_next = temporal_context.get('next', {}).get('severity_z_score', 0)
            if z_next > 2.0:
                lage_bericht += " Auch am Folgetag wurden Anomalien gemessen."

        context_xml = "\n".join([
            f"<quelle id='{i+1}' titel='{s['title']}'>\n{s['content']}\n</quelle>"
            for i, s in enumerate(validated_sources)
        ])
        
        system_prompt = f""" 
        Du bist leitender Datenanalyst für die systemische Rekonstruktion von Bahnbetriebslagen. 
        Deine Aufgabe ist es, statistische Anomalien im Fernverkehr (ICE/IC) ausschließlich anhand verifizierter Kontextdokumente logisch zu begründen. 
        Du agierst streng datengetrieben, wertneutral und präzise.
        """        
        
        user_prompt = f"""
        ### SYSTEM-AUFTRAG: KAUSALANALYSE EINER NETZANOMALIE
        - Untersuchungstag: {self._format_date(date_str)}
        - STRIKTES ZIELJAHR: {year_target} (Verwende niemals Quellen oder Ereignisse, die in einem anderen Jahr stattfinden!)
        
        ### STATISTISCHE AUSGANGSDATEN
        - Gemessene Durchschnittsverspätung im Netz: {delay_value:.2f} Minuten
        - Globaler Erwartungswert (Durchschnittliche Netzlast): {avg_delay_global:.2f} Minuten
        - Festgelegter Anomalie-Grenzwert (Threshold): {threshold_val:.2f} Minuten
        
        ### ZEITLICHER NETZKENTEXT
        {lage_bericht}
        
        ### VERIFIZIERTE KONTEXTDOKUMENTE (RAG-ZULAUF)
        <context>
        {context_xml}
        </context>

        ### STRUKTURELLER ARBEITSAUFTRAG
        Generiere die Ursachenanalyse streng strukturiert entlang der folgenden drei Abschnitte. Halte dich exakt an das vorgegebene Muster.

        ### 1. Ursache
        - Titel: <Generiere hier einen kurzen, prägnanten Titel des primären Störungsgeschehens>
        - Kategorie: <Wähle hier exakt eine passende Kategorie aus: METEOROLOGISCHE EINFLÜSSE | TECHNOLOGISCHE EINFLÜSSE | EXTERNE EINFLÜSSE | KUMULATIVE EINFLÜSSE>

        ### 2. Analyse
        - Rekonstruiere die exakte Kausalkette für den Tag {self._format_date(date_str)} im Jahr {year_target}.
        - Zitiere relevante Fakten (z.B. Streckensperrungen, Signalstörungen oder Teilausfälle) unter Angabe der Quellen-ID (z.B. [Quelle 1]).

        ### 3. Plausibilitätsprüfung
        - Prüfe kritisch, ob die Dokumente die Verspätung von {delay_value:.2f} Minuten für das Jahr {year_target} logisch erklären.
        - Falls die Dokumente inhaltlich oder zeitlich nicht exakt zum Untersuchungstag passen, deklariere dies als informationelle Lücke.
        """
        
        start_analysis = time.time()
        try:
            response = ollama.chat(
                model=self.model_name, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'num_ctx': context_size, 'temperature': 0.1}
            )
            analysis_dur = round(time.time() - start_analysis, 2)
            
            return {
                "answer": response['message']['content'],
                "sources": [{"title": s['title'], "url": s['url']} for s in validated_sources],
                "timings": {
                    "filtering": filtering_dur,
                    "final_analysis": analysis_dur
                }
            }
        except Exception as e:
            analysis_dur = round(time.time() - start_analysis, 2)
            return {
                "answer": f"Fehler bei der KI-Analyse: {e}", 
                "sources": [],
                "timings": {
                    "filtering": filtering_dur,
                    "final_analysis": analysis_dur
                }
            }