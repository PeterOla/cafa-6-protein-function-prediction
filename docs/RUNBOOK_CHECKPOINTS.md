# CAFA6 Runbook — Resumable Runs via Kaggle Dataset Checkpoints

This repo uses a **single canonical checkpoint store**: a Kaggle Dataset.

The all-in-one notebook ([notebooks/Colab_04_all_in_one.ipynb](../notebooks/Colab_04_all_in_one.ipynb)) does two things automatically:
- `STORE.pull()` at startup (hydrates `WORK_ROOT` so you can resume anywhere)
- `STORE.push(stage, paths)` at milestones (publishes a new dataset version)

No other publishing flow should be used.

---
## 0) One-time setup (applies everywhere)

1) Choose your checkpoint dataset:
- Set `CAFA_CHECKPOINT_DATASET_ID=<kaggle-username>/<dataset-slug>`
- Set `CAFA_CHECKPOINT_DATASET_TITLE` if you want to override the title (defaults to `CAFA6 Checkpoints`)

Where do you find `CAFA_CHECKPOINT_DATASET_ID`?
- It’s just the Kaggle Dataset identifier in the URL:
   - Open your dataset in Kaggle → the URL looks like `https://www.kaggle.com/datasets/<username>/<dataset-slug>`
   - Your `CAFA_CHECKPOINT_DATASET_ID` is `<username>/<dataset-slug>`
- If you don’t have a dataset yet, pick a new slug (e.g. `cafa6-checkpoints`) and set:
   - `CAFA_CHECKPOINT_DATASET_ID=<your-username>/cafa6-checkpoints`
   - On first publish, the notebook will attempt to create/version that dataset via the Kaggle API.

2) Provide auth (API-only):
- Kaggle CLI publishing/downloading requires Kaggle API credentials (from `kaggle.json`):
   - Set `KAGGLE_USERNAME=<your-username>` and `KAGGLE_KEY=<your-key>` via **Kaggle Secrets** or **Colab secrets**
   - The notebook will try, in order: env var → Kaggle Secrets → Colab `userdata`.

Back-compat:
- `KAGGLE_API_TOKEN` is only supported if it contains the full `kaggle.json` JSON payload or `username:key`.

Note: `CAFA_CHECKPOINT_DATASET_ID` is resolved the same way (env var → Kaggle Secrets → Colab `userdata`).

3) Controls (defaults shown):
- `CAFA_CHECKPOINT_PULL=1` (default) to pull on startup
- `CAFA_CHECKPOINT_PUSH=1` (default) to publish milestones
- `CAFA_CHECKPOINT_REQUIRED=1` (default) to fail-fast if checkpoints are not accessible (set to `0` for best-effort warning-only pulls)

---
## 1) Kaggle (recommended final consumer)

Goal: run from a fresh Kaggle runtime, resume automatically.

Steps:
1) Create a Kaggle Notebook with GPU enabled.
2) Add inputs:
   - The CAFA-6 competition dataset
   - Your checkpoint dataset `<user>/<slug>` (recommended: attach as Input so `STORE.pull()` uses the mount)
3) Add Secrets:
   - `KAGGLE_USERNAME` = `<your-username>`
   - `KAGGLE_KEY` = `<your-key>`
   - `CAFA_CHECKPOINT_DATASET_ID` = `<user>/<slug>`
4) Open and run [notebooks/Colab_04_all_in_one.ipynb](../notebooks/Colab_04_all_in_one.ipynb).

Notes:
- If the checkpoint dataset is attached as an input, `STORE.pull()` uses the mounted copy (fast, no network).
- If it is not attached, `STORE.pull()` will try the Kaggle API download (needs internet + auth).
- If you hit **HTTP 403 Forbidden** on pull, it almost always means the dataset is **private** or not shared with the Kaggle account running the notebook. The pragmatic fix is to **attach the checkpoint dataset as a Notebook Input**, or make/share the dataset.

Pip note (Kaggle):
- Kaggle images come with many preinstalled packages. Blindly running `pip install -r requirements.txt` can upgrade core deps and produce resolver “dependency conflicts” warnings.
- The all-in-one notebook avoids upgrading core packages on Kaggle by default; set `CAFA_FORCE_PIP=1` only if you explicitly want a full `requirements.txt` install.

---
## 2) Colab (good for heavy GPU stages)

Goal: do the expensive embedding stages on stronger GPUs, then publish and resume on Kaggle.

Minimum requirements:
- You must have the repo contents available (including `Train/`, `Test/`, `IA.tsv`, `sample_submission.tsv`).

Steps:
1) Get the repo + data into Colab (pick one):
   - Mount Drive and copy the repo folder there
   - Or `git clone` your repo if it contains the competition files
2) Set secrets/env vars:
   - `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `CAFA_CHECKPOINT_DATASET_ID` (either as env vars or Colab `userdata`).
3) Run [notebooks/Colab_04_all_in_one.ipynb](../notebooks/Colab_04_all_in_one.ipynb).

Recommended split (fastest wall-clock):
- Colab: run the embedding cells (T5/ESM2) → they checkpoint via `STORE.push(...)`
- Kaggle: resume and run the remaining training/stacking/submission steps

---
## 3) Local (Windows)

Goal: iterate/debug without Kaggle constraints, and publish checkpoints via the Kaggle API when needed.

Steps:
1) Ensure the repo root contains `Train/`, `Test/`, `IA.tsv`, `sample_submission.tsv`.
2) Create a Python environment and install requirements:
   - `pip install -r requirements.txt`
3) Set env vars (PowerShell example):
   - `$env:CAFA_CHECKPOINT_DATASET_ID = "<user>/<slug>"`
   - `$env:KAGGLE_USERNAME = "<your-username>"`
   - `$env:KAGGLE_KEY = "<your-key>"`
4) Run the notebook.

Tip:
- Keep `CAFA_CACHE_ROOT` on a fast disk; it is not published.

---
## How resume actually works (mental model)

- `WORK_ROOT` is the **published artefact root**.
- `CACHE_ROOT` is **ephemeral cache** (HF/Torch/etc) and is intentionally excluded from publishing.
- Each milestone cell:
  - checks whether required files exist under `WORK_ROOT`
  - only recomputes if missing
  - then publishes that stage via `STORE.push(stage, required_paths)`

If you want a clean rerun, delete the relevant files under `WORK_ROOT` (or point `CAFA_WORK_ROOT` somewhere new).

---
## Cells to run (by environment) — in order

This is the concrete “what to run where” mapping for [notebooks/Colab_04_all_in_one.ipynb](../notebooks/Colab_04_all_in_one.ipynb).

### Kaggle (recommended finishing environment)

Required “resume + finish” run (GPU on):
- `# CELL 02` (Install dependencies)
- `# CELL 03` (Setup + `STORE.pull()`)
- `# CELL 04` (Parse/structuring + hierarchy)
- `# CELL 05` (Inline: EntryID→text corpus builder)
- `# CELL 06` (Run: build EntryID→text corpus)
- `# CELL 07` (Inline: embeddings generator; required by TF‑IDF)
- `# CELL 08` (Run: TF‑IDF embeddings)
- `# CELL 09` + `# CELL 10` + `# CELL 10b` (external GOA + propagation + artefact diagnostics)
- `# CELL 11` (T5 embeddings)
- `# CELL 13` + `# CELL 14` + `# CELL 15` + `# CELL 16` (Level‑1 → stacker → post‑processing → submission)

Not required for a valid CAFA submission:
- `# CELL 17` (Free text prediction)

Notes:
- `# CELL 01` is only needed if you’re using the notebook to clone/update the repo inside Kaggle.

### Colab (recommended heavy GPU stages)

Run embeddings and publish checkpoints, then stop:
- `# CELL 02` (Install dependencies)
- `# CELL 03` (Setup + `STORE.pull()`)
- `# CELL 11` (T5 embeddings)
- `# CELL 12` (ESM2 embeddings)

Note:
- The EntryID→text corpus + TF‑IDF + external GOA steps are treated as required for the strict runbook path; do them on Kaggle.

### Local (Windows) — debugging/iteration

Smallest sensible “sanity run” (CPU is fine):
- `# CELL 02` (Install dependencies)
- `# CELL 03` (Setup + `STORE.pull()`)
- `# CELL 04` (Parse/structuring)
- `# CELL 10b` (Artefact manifest diagnostics)

If you want to publish from local:
- Ensure `KAGGLE_USERNAME` and `KAGGLE_KEY` are set, then run the stage cells you care about; they will call `STORE.push(...)`.

---
## Dependencies (what depends on what)

Key idea: there is no magical background “dataset population”. The checkpoint dataset only gains files when *you* run the producing cells and they publish via `STORE.push(...)`.

### Milestones (files) and the cells that produce them

- **Stage 01: parsed core** (`WORK_ROOT/parsed/*`) → produced by `# CELL 04`
   - Required by: everything downstream (`# CELL 05` onwards)
- **Stage 02: entryid→text corpus** (`external/entryid_text.tsv`) → produced by `# CELL 06`
   - Required by: `# CELL 08`
- **Stage 03: TF‑IDF text embeddings** (`features/train_embeds_text.npy`, `features/test_embeds_text.npy`, `features/text_vectorizer.joblib`) → produced by `# CELL 08`
   - Required for the strict runbook path
- **Stage 04: external GOA priors** (`external/prop_train_no_kaggle.tsv.gz`, `external/prop_test_no_kaggle.tsv.gz`) → produced by `# CELL 10`
   - Required for the strict runbook path
- **Stage 05: T5 embeddings** (`features/train_embeds_t5.npy`, `features/test_embeds_t5.npy`) → produced by `# CELL 11`
   - Required by: `# CELL 13` (Level‑1) (hard requirement)
- **Stage 06: ESM2 embeddings** (`features/train_embeds_esm2.npy`, `features/test_embeds_esm2.npy`) → produced by `# CELL 12`
   - Not required (Level‑1 uses it if present)
- **Stage 07: Level‑1 predictions** (`features/oof_pred_*.npy`, `features/test_pred_*.npy`, `features/top_terms_1500.json`) → produced by `# CELL 13`
   - Required by: `# CELL 14` / `# CELL 15`
- **Stage 08: stacker predictions** (`features/test_pred_gcn.npy`, `features/top_terms_1500.json`) → produced by `# CELL 14` / `# CELL 15`
   - Required by: `# CELL 16`
- **Stage 09: submission** (`submission.tsv`) → produced by `# CELL 16`

### What you can run before the checkpoint dataset has anything in it

If your checkpoint dataset is empty (first ever run), you **do not wait** — you simply run producers in order:
- Core minimum path: `# CELL 02` → `# CELL 03` → `# CELL 04` → `# CELL 11` → `# CELL 13` → `# CELL 14`/`# CELL 15` → `# CELL 16`
- Option B strict path: (minimum path) + `# CELL 05` → `# CELL 06` → `# CELL 07` → `# CELL 08` + `# CELL 10`

### “I published on Colab, but Kaggle can’t see it yet”

This is usually an input-mount/versioning issue:
- If Kaggle has your checkpoint dataset attached as an **input**, it will only see the version available at notebook start; restart the Kaggle session to pick up the latest.
- If you want to always pull the latest without restarting, don’t attach the dataset as an input and let `STORE.pull()` download via the Kaggle API (requires internet + Kaggle creds).
