"""
Pobiera (snapshot_download) modele Whisper do obrazu
bez ich ładowania do RAM-u.

Uruchamiane w Dockerfile (patrz sekcja 5).
"""

import logging
import os, sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading Models: %s",os.getenv("WHISPER_MODELS", "tiny"))

DEST = Path("/models")
MODELS = [m.strip() for m in os.getenv("WHISPER_MODELS", "tiny").split(",") if m]

if not MODELS:
    sys.exit("Brak modeli w WHISPER_MODELS !")
