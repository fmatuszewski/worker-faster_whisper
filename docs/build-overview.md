# Whisper Container Build & Deployment - Quick Guide

## 1. Why two image tiers?

* **Core images** (`*-core`) - carry the heavy Whisper model weights. They rebuild **only** when they are missing in Docker Hub.
* **Child images** (`tiny` `small` `large`) - contain only your Python runtime code (scripts). They rebuild automatically on every commit, keeping startup tiny and CI fast.

## 2. Repo layout

```text
/               - root of the repo
├ Dockerfile.core     - builds model‑heavy base images
├ Dockerfile.child    - builds slim runtime images
├ builder/
│   └ fetch_models.py - downloads only the models listed in $WHISPER_MODELS
└ scripts/            - your app code (main.py, utils…)
```

## 3. GitHub Actions workflow (`.github/workflows/build.yml`)

| Job       | What it does                                                                             | When it runs                                    |
| --------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **core**  | Builds `tiny-core`, `small-core`, `large-core` **iff** the tag is missing in Docker Hub. | On every push (but exits early if image exists) |
| **child** | Always builds `tiny`, `small`, `large`, based on the matching core tag.                  | Needs **core**; runs on every push              |

### Key mechanics

* Matrix strategy passes the model list (`WHISPER_MODELS`) to **Dockerfile.core**
* Existence check: `docker manifest inspect` returns 0 if the tag is already in the registry
* Cache layers are kept in `worker-whisper:cache` so subsequent builds reuse downloaded weights

## 4. Secrets required

| Secret            | Purpose                                |
| ----------------- | -------------------------------------- |
| `DOCKERHUB_USER`  | Your Docker Hub username/org           |
| `DOCKERHUB_TOKEN` | Access token with **write** permission |

## 5. Typical developer actions

| Task               | How to do it                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| Ship new code      | `git push` → only child images rebuild                                                              |
| Force‑rebuild core | Delete the `*-core` tag in Docker Hub **or** run the workflow manually via *Actions → Run workflow* |
| Add extra model    | Edit `strategy.matrix` in **core** job and adjust `WHISPER_MODELS` list + tag name                  |

## 6. Runtime

* Containers run exactly the weight set bundled in their core image; no `git pull` or SSH keys needed at runtime
* Entry‑point defined in **Dockerfile.child** (`ENTRYPOINT ["python", "main.py"]`). Change it if your script name differs

## 7. Troubleshooting

| Symptom                                  | Likely cause                                             | Fix                                  |
| ---------------------------------------- | -------------------------------------------------------- | ------------------------------------ |
| Child image fails: missing `/models/...` | Core tag was deleted & not yet rebuilt                   | Rerun workflow or wait for next push |
| Core build OOM‑kills                     | Split `WHISPER_MODELS` list or raise runner memory in CI |                                      |

## 8. Goals

*   Create a CI/CD pipeline for building and deploying Whisper container images.
*   Build core images only when they are missing in Docker Hub.
*   Build child images on every commit.
*   Allow for building core images with different sets of models.
*   Allow for building child images based on different core images.

## Appendix: Updated Implementation Details

### Dockerfile.core

The `Dockerfile.core` now accepts an argument `WHISPER_MODELS` to specify which models to download. This allows for building core images with different sets of models.

```dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# ... (Base OS and Python setup) ...

ARG WHISPER_MODELS=tiny
ENV WHISPER_MODELS=${WHISPER_MODELS}

COPY builder/fetch_models.py /tmp/fetch_models.py
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python /tmp/fetch_models.py && rm /tmp/fetch_models.py

LABEL whisper.core.models=${WHISPER_MODELS}
```

### Dockerfile.child

The `Dockerfile.child` now uses arguments `DOCKERHUB_USER` and `BASE_TAG` to dynamically specify the base image. This allows for building child images based on different core images.

```dockerfile
ARG DOCKERHUB_USER
ARG BASE_TAG

FROM ${DOCKERHUB_USER}/worker-whisper:${BASE_TAG}

WORKDIR /app
COPY scripts/ .

ENTRYPOINT ["python", "main.py"]
```

### fetch_models.py

The `fetch_models.py` script now reads the `WHISPER_MODELS` environment variable to determine which models to download.

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path("/models")
MODELS = [m.strip() for m in os.getenv("WHISPER_MODELS", "tiny").split(",") if m]

if not MODELS:
    raise ValueError("No models specified in WHISPER_MODELS environment variable!")

DEST.mkdir(parents=True, exist_ok=True)
for name in MODELS:
    snapshot_download(
        repo_id=f"openai/whisper-{name}",
        local_dir=str(DEST / name),
        local_dir_use_symlinks=False,
        resume_download=True
    )