"""
Pobiera (snapshot_download) modele Whisper do obrazu
bez ich ładowania do RAM-u.

Uruchamiane w Dockerfile (patrz sekcja 5).
"""

from pathlib import Path
from huggingface_hub import snapshot_download

DEST_DIR = Path("/models")        # możesz zmienić
MODELS = ["tiny", "base", "medium", "large-v2"]

def download(model_name: str) -> None:
    repo_id = f"openai/whisper-{model_name}"
    out_dir = DEST_DIR / model_name
    print(f"⬇️  {repo_id}  →  {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,   # bez dyskowych sztuczek
        resume_download=True            # wznawia po przerwaniu
    )

def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        download(name)
    print("✅ Wszystkie modele pobrane.")

if __name__ == "__main__":
    main()
