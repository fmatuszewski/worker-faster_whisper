"""
Pobiera (snapshot_download) modele Whisper do obrazu
bez ich ładowania do RAM-u.

Uruchamiane w Dockerfile (patrz sekcja 5).
"""

import logging
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading Models: %s",os.getenv("WHISPER_MODELS", "tiny"))

DEST = Path("/models")
MODELS = [m.strip() for m in os.getenv("WHISPER_MODELS", "tiny").split(",") if m]

if not MODELS:
    sys.exit("Brak modeli w WHISPER_MODELS !")

DEST.mkdir(parents=True, exist_ok=True)
for name in MODELS:
    snapshot_download(
        repo_id=f"openai/whisper-{name}",
        local_dir=str(DEST / name),
        local_dir_use_symlinks=False,
        resume_download=True
    )
