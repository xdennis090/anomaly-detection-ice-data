"""Zentrale Konfigurationsdatei für die unterstützten lokalen Sprachmodelle.
Dieses Modul definiert die verfügbaren Large Language Models (LLMs) 
sowie deren spezifische Modell-IDs und maximale Kontextfenster-Größen 
für die dynamische Orchestrierung innerhalb der Pipeline.
"""

from typing import Dict, Any

SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "1. Llama 3.1 8B": {
        "id": "llama3.1",
        "ctx": 8192
    },
    "2. Mistral Nemo 12B": {
        "id": "mistral-nemo",
        "ctx": 12288
    },
    "3. Gemma 2 27B": {
        "id": "gemma2:27b",
        "ctx": 16384
    },
    "4. Command R 35B": {
        "id": "command-r",
        "ctx": 32768
    },
    "5. Qwen 2.5 32B": {
        "id": "qwen2.5:32b",
        "ctx": 32768
    }
}