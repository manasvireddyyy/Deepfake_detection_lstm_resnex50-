# Developer Guide — Deepfake Detection (ResNeXt + LSTM)

This developer guide is written for contributors and maintainers. It explains repository layout, how to set up a development environment, how inference and model selection work, how to train and preprocess data (notebooks included), and recommended refactors and tests to keep the codebase maintainable.

## Contents
- Overview & goals
- Repo layout and key files
- Local setup and run commands
- Model architecture and where code lives
- Model selection, adding and forcing models
- Preprocessing & training (notebooks)
- Debugging and common issues
- Recommended refactor and testing plan
- CI & deployment notes

---

## Overview & goals

This project detects deepfake videos using a ResNeXt50 backbone to extract per-frame spatial features and an LSTM to model temporal patterns across frames. The app is a Django web application that lets users upload a video, choose a sequence length, and view a `REAL` / `FAKE` prediction with a confidence score and saved visualizations.

Primary developer goals:
- Keep the inference path simple and debuggable.
- Keep model artifacts (`.pt`) in `models/` and make selection deterministic.
- Keep preprocessing reproducible via the notebooks in `Model Training/`.

## Repository layout (important files)

- `manage.py` — Django entrypoint.
- `project_settings/` — Django settings and `urls.py`.
- `ml_app/`
  - `views.py` — core inference flow, model class, dataset helper, and `predict_page()` handler.
  - `forms.py` — `VideoUploadForm` (file + `sequence_length`).
  - `templates/` — UI templates (`index.html`, `predict.html`, etc.).
- `models/` — directory to store `.pt` model files (PyTorch checkpoints).
- `uploaded_videos/`, `uploaded_images/` — runtime artifacts for uploaded files and preprocessing outputs.
- `Model Training/` — Jupyter notebooks for preprocessing and training (`preprocessing.ipynb`, `Model_and_train_csv.ipynb`).
- `requirements.txt` — pinned dependencies used during development.
- `APP_DOCUMENTATION.md` — user-facing documentation (overview, usage).

When making changes, prefer small, targeted modifications and keep ML and web concerns separated where possible.

## Local setup (macOS, zsh)

1. Activate the project's virtual environment (there's a `venv/` in repo):

```bash
source /Users/margamsairam/Desktop/Deepfake-detection-using-Deep-Learning-ResNext-and-LSTM-/venv/bin/activate
```

2. (Optional) Install dependencies if you're missing packages:

```bash
pip install -r requirements.txt
```

3. Run the Django development server:

```bash
python3.10 manage.py runserver
```

4. Open `http://127.0.0.1:8000/` and use the UI.

Notes:
- If `face_recognition` installation fails on macOS, install system deps first: `brew install cmake pkg-config`, then `pip install dlib` and `pip install face-recognition`.

## Model code and architecture

- The model is implemented in `ml_app/views.py` as `class Model(nn.Module)`.
- Architecture summary:
  - Backbone: ResNeXt50 (`resnext50_32x4d`) truncated (`children()[:-2]`) to produce feature maps (2048 channels).
  - Pooling: `AdaptiveAvgPool2d(1)` → 2048-d per-frame vector.
  - Temporal: `nn.LSTM(latent_dim=2048, hidden_dim=2048, ...)` processes the sequence.
  - Classifier: `nn.Linear(2048, 2)` outputs logits for `FAKE` / `REAL`.

Why this lives in `views.py`:
- The repo is small and the author placed model and dataset helpers there to keep everything in one file for quick prototyping.

Where to extract if you refactor:
- `ml_app/model.py` — model class and load/save helpers.
- `ml_app/dataset.py` — `video_dataset`, `validation_dataset` classes.
- `ml_app/inference.py` — `predict`, `plot_heat_map`, `get_accurate_model`.

## Model selection logic

- Current selection function: `get_accurate_model(sequence_length)` in `ml_app/views.py`.
  - It lists `models/*.pt` and looks for filenames where the 4th token (index 3 after `.split('_')`) equals the requested `sequence_length`.
  - If multiple models match, it chooses the one with the maximum accuracy token (index 1 in filename). The existing code compares accuracy tokens as strings — convert to `float()` for numeric comparison.

### Example filename convention

Keep model filenames in this pattern for automatic selection:

```
model_<accuracy>_acc_<sequence>_frames_<...>.pt
```

Example: `model_95_acc_40_frames_FF_data.pt` → accuracy 95, sequence length 40.

### Forcing a specific model

If you want to force a particular checkpoint, edit `predict_page()` and replace the `model_name` with a fixed filepath or set `path_to_model`:

```python
path_to_model = os.path.join(settings.PROJECT_DIR, 'models', 'model_95_acc_40_frames_FF_data.pt')
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
```

Recommendation: replace the hard-coded override in `predict_page()` with a call to `get_accurate_model(sequence_length)` and log the chosen model.

## Preprocessing & Training (notebooks)

- Use the notebooks in `Model Training/` for dataset creation and training.
  - `preprocessing.ipynb` — extracts faces from videos and writes face-only videos (112x112) into a Drive folder (Colab-oriented).
  - `Model_and_train_csv.ipynb` — builds `video_dataset`, defines the model, training loop, and evaluation tools.

Key points when running notebooks:
- Notebooks expect dataset paths typically on Google Drive. Update paths to local folders if running locally.
- Use `train_transforms` and ImageNet normalization used in inference (`112 x 112`, mean/std values) to keep preprocessing consistent.

## Debugging and common issues

- Model not switching to requested sequence length:
  - Check `request.session['sequence_length']` is set after uploading (see `index()` in `views.py`).
  - Remove any hard-coded `model_name` override in `predict_page()` (repo currently has one).
  - Run the quick check from project root to confirm selection:

```bash
python3.10 - <<'PY'
from ml_app import views
print('Model for 40 ->', views.get_accurate_model(40))
PY
```

- Torchvision deprecation warnings (`pretrained=True`): use the `weights=` API or keep the fallback for older versions.

- `face_recognition` errors: ensure `dlib` and its build deps installed first; in Colab it's usually straightforward.

- No faces detected on a video: try clearer frames, or adjust padding and face detection parameters. The inference path uses `face_recognition.face_locations()`.

## Recommended refactor & unit test plan

Goal: separate responsibilities and make small functions testable. Suggested steps:

1. Extract model, dataset, and inference helpers into modules:
   - `ml_app/model.py` — `Model`, `load_model(path, device)`.
   - `ml_app/dataset.py` — `video_dataset`, `validation_dataset`, utility `frame_extract`.
   - `ml_app/inference.py` — `predict`, `plot_heat_map`, `get_accurate_model`.

2. Add unit tests for pure-Python functions with pytest:
   - Test `get_accurate_model()` behavior with a temporary directory and fake filenames.
   - Test dataset behavior with a small synthetic video or generated frames (mock `cv2.VideoCapture`).
   - Test filename/label lookup and CSV parsing logic.

3. Add a minimal CI pipeline (GitHub Actions) that runs linting and the unit tests.

4. Add logging rather than print statements in `predict_page()` and `get_accurate_model()` to aid debugging in production.

## Quick unit test example (pytest)

Create `tests/test_model_selection.py`:

```python
import os
from ml_app.views import get_accurate_model

def test_get_accurate_model_tmpdir(tmp_path, monkeypatch):
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    (models_dir / 'model_90_acc_60_frames_final_data.pt').write_text('x')
    (models_dir / 'model_95_acc_40_frames_FF_data.pt').write_text('x')
    # monkeypatch project settings to point to tmp_path
    import django.conf
    monkeypatch.setattr(django.conf.settings, 'PROJECT_DIR', str(tmp_path))
    res = get_accurate_model(40)
    assert 'model_95_acc_40_frames_FF_data.pt' in res
```

Note: small adjustments may be required to import `settings` or to design `get_accurate_model()` to accept the models directory path for testability.

## CI & deployment notes

- For production, do not use DEBUG mode.
- Secure uploads: add cleanup cron job or TTL for files in `uploaded_videos/` and `uploaded_images/`.
- Consider serving model inference through a separate worker (FastAPI or a microservice) if load increases.

## Next recommended tasks (PRs)

1. Refactor `ml_app/views.py` into separate modules and add unit tests.
2. Replace hard-coded model path with robust `get_accurate_model()` and logging.
3. Add an admin or small upload view to safely add model files via the web UI.
4. Add a lightweight CI pipeline that runs tests.

---

If you want, I can implement any of the suggested PRs: refactor code into modules (A), fix and harden the model selection (B), or add a model management UI (C). Tell me which and I'll start. 
