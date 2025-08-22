# ✅ TESTING.md — Mini‑SOAR Project

This document explains how to test the **Mini‑SOAR** repository across units, integrations, models, and UI. It is aligned with the project’s current code (Streamlit app, PyCaret training, and GenAI prescriptions).

---

## 1) Goals & Scope
- Verify **classification** and **clustering** behavior on the synthetic feature set.
- Confirm **gating logic** in the Streamlit app: clustering & GenAI only trigger on malicious verdicts.
- Validate **GenAI provider routing** without making live API calls.
- Ensure **reproducibility** and **non‑interactive CI** execution.

---

## 2) Environment
- Python **3.9+** (3.11 works in logs/containers)
- Core deps: `pytest`, `pytest-cov` plus repo requirements
- Headless plotting is already configured in training via `matplotlib.use("Agg")`.

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

> Tip: Use a virtualenv/conda. In CI, run tests before starting Streamlit to keep jobs fast.

---

## 3) Repository Layout 
```
mini-soar/
├── app.py
├── train_model.py
├── genai_prescriptions.py
├── models/                  # created by training
├── data/                    # created by training
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_train_model.py
    ├── test_app_logic.py
    └── test_genai.py
```

Create the `tests/` directory with the files below.

---


Happy testing! 🧪

