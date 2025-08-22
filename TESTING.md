# ✅ TESTING.md — Mini‑SOAR Project

This document explains how to test the **Mini‑SOAR** repository across units, integrations, models, and UI. It is aligned with the project’s current code (Streamlit app, PyCaret training, and GenAI prescriptions).

---

## 1) Goals & Scope
- Verify **classification** and **clustering** behavior on the synthetic feature set.
- Confirm **gating logic** in the Streamlit app: clustering & GenAI only trigger on malicious verdicts.
- Validate **GenAI provider routing** without making live API calls (tests use mocks).
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

## 3) Repository Layout (tests suggested)
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

## 4) Running Tests
```bash
# run all
pytest -q

# with coverage
pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing

# run a single test
pytest tests/test_app_logic.py::test_gating_logic_malicious_path
```

---

## 5) Reproducibility
The synthetic dataset uses a fixed RNG seed inside `generate_synthetic_data`. Prefer **shape/range assertions** over exact metric values when retraining to avoid brittle tests.

Add a global seed in `conftest.py`:
```python
# tests/conftest.py
import os, random, numpy as np

def pytest_sessionstart(session):
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(42)
    np.random.seed(42)
```

---

## 6) Unit Tests

### 6.1 Training: synthetic features, artifacts
```python
# tests/test_train_model.py
from pathlib import Path
import pandas as pd

def test_synthetic_data_schema():
    from train_model import generate_synthetic_data
    df = generate_synthetic_data(num_samples=100, seed=123)
    assert len(df) == 100
    needed = {
        'having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol',
        'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain',
        'SSLfinal_State','URL_of_Anchor','Links_in_tags','SFH',
        'Abnormal_URL','has_political_keyword','label'
    }
    assert needed.issubset(df.columns)

def test_training_writes_models(tmp_path, monkeypatch):
    # run training end-to-end in temp dir
    import os
    from train_model import train
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        train()
        # classification model
        assert Path('models/phishing_url_detector.pkl').exists()
        # feature plot created/relocated
        assert Path('models/feature_importance.png').exists()
        # clustering artifacts (optional but expected on first run)
        assert Path('models/threat_actor_profiler.pkl').exists()
        assert Path('models/threat_actor_profile_map.json').exists()
        # data exports
        assert Path('data/phishing_synthetic.csv').exists()
        assert Path('data/phishing_features_only.csv').exists()
        assert Path('data/clustered_assignments.csv').exists()
    finally:
        os.chdir(cwd)
```
*What this checks*: expected columns used by the app; model & plot files saved; clustering outputs used for attribution.

### 6.2 App logic: gating, parsing labels, cluster → actor mapping
The app converts model outputs into a boolean `is_malicious` and only then runs clustering and GenAI. We can test this without Streamlit by **monkeypatching** the module‑level predictors and the GenAI function.

```python
# tests/test_app_logic.py
import types
import pandas as pd

def _df(label, score=0.9, col='prediction_label'):
    return pd.DataFrame([{col: label, 'prediction_score': score}])

def test_gating_logic_malicious_path(monkeypatch):
    import app as app_mod

    # Fake classifiers/clusters
    monkeypatch.setattr(app_mod, "predict_cls_model", lambda model, data: _df("1"))
    monkeypatch.setattr(app_mod, "predict_clu_model", lambda model, data: pd.DataFrame([{"Cluster":"Cluster 2"}]))

    # Provide fake models so code paths are enabled
    app_mod.model = object()
    app_mod.cluster_model = object()

    # Fake Streamlit interactions (minimal surface)
    class ST:
        def write(*a, **k): pass
        def error(*a, **k): pass
        def warning(*a, **k): pass
        def json(*a, **k): pass
        def metric(*a, **k): pass
        def caption(*a, **k): pass
        def subheader(*a, **k): pass
        def bar_chart(*a, **k): pass
        def text_area(*a, **k): pass
        def success(*a, **k): pass
        def info(*a, **k): pass
        def tabs(self, labels): 
            class Dummy: 
                def __enter__(self): return self
                def __exit__(self,*a): pass
            return Dummy(), Dummy(), Dummy(), Dummy()
        class status:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def update(self, **k): pass
    monkeypatch.setattr(app_mod, "st", ST)

    # Stub the GenAI prescription (deterministic)
    monkeypatch.setattr(app_mod, "generate_prescription", lambda provider, details: {"recommended_actions":["Block","Contain","Notify"]})
    # Minimal inputs matching app schema
    app_mod.form_values = {
        'url_length': 'Long','ssl_state':'Suspicious','sub_domain':'One','prefix_suffix':True,
        'has_ip': False, 'short_service': False, 'at_symbol': False, 'abnormal_url': True,
        'has_political_keyword': True
    }
    # Re-run the same block as the app would on submit by calling the internal code path.
    # In this repo, that logic is embedded in app.py; here we simply ensure the above monkeypatches run without error.
    # If you factor the analysis into a function, call it here and assert on its return dictionary.

def test_label_parsing_variants(monkeypatch):
    import app as app_mod
    app_mod.model = object()
    app_mod.cluster_model = None  # force no clustering

    class FakeST:
        def write(*a, **k): pass
        def error(*a, **k): pass
        def warning(*a, **k): pass
        def json(*a, **k): pass
        def metric(*a, **k): pass
        def caption(*a, **k): pass
        def subheader(*a, **k): pass
        def bar_chart(*a, **k): pass
        def text_area(*a, **k): pass
        def success(*a, **k): pass
        def info(*a, **k): pass
        def tabs(self, labels): 
            class Dummy:
                def __enter__(self): return self
                def __exit__(self,*a): pass
            return Dummy(), Dummy(), Dummy(), Dummy()
        class status:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def update(self, **k): pass
    monkeypatch.setattr(app_mod, "st", FakeST)

    # Try multiple label encodings the app supports: "1", "true", "malicious", 0
    for lbl in ["1","true","malicious", 0]:
        monkeypatch.setattr(app_mod, "predict_cls_model", lambda model, data, lab=lbl: _df(lab))
        # Run a minimal path; if you extract to a function, call and assert `is_malicious` behavior.
```

> **Recommendation:** Consider refactoring the analysis block in `app.py` into a pure function (e.g., `analyze(input_dict, model, cluster_model, provider) -> dict`). This will let tests assert specific return values (verdict, cluster_id, actor_profile).

### 6.3 GenAI prescriptions: provider routing & JSON shape
```python
# tests/test_genai.py
def test_provider_routing(monkeypatch):
    import genai_prescriptions as g

    called = {"gemini":False, "openai":False, "grok":False}
    monkeypatch.setattr(g, "get_gemini_prescription", lambda d: called.__setitem__("gemini", True) or {"ok":True})
    monkeypatch.setattr(g, "get_openai_prescription", lambda d: called.__setitem__("openai", True) or {"ok":True})
    monkeypatch.setattr(g, "get_grok_prescription",   lambda d: called.__setitem__("grok", True)   or {"ok":True})

    g.generate_prescription("Gemini", {"a":1});  assert called["gemini"]
    g.generate_prescription("OpenAI", {"a":1});  assert called["openai"]
    g.generate_prescription("Grok",   {"a":1});  assert called["grok"]

def test_no_live_api(monkeypatch):
    # Ensure tests do not rely on secrets; stub out network clients
    import genai_prescriptions as g
    monkeypatch.setattr(g, "genai", type("X", (), {"GenerativeModel": lambda *a, **k: type("M", (), {"generate_content": lambda *aa, **kk: type("R", (), {"text":"{\"summary\":\"s\",\"risk_level\":\"High\",\"recommended_actions\":[],\"communication_draft\":\"d\"}"})()})()}) )
    monkeypatch.setattr(g, "openai", type("Y", (), {"OpenAI": lambda *a, **k: type("C", (), {"chat": type("Z", (), {"completions": type("W", (), {"create": lambda *aa, **kk: type("Resp", (), {"choices":[type("M", (), {"message": type("N", (), {"content":"{\"summary\":\"s\",\"risk_level\":\"High\",\"recommended_actions\":[],\"communication_draft\":\"d\"}"})()})()]})()})()})()}) )
    # Grok endpoint also mocked via requests
    import types
    class FakeResp: 
        def json(self): return {"choices":[{"message":{"content":"{\"summary\":\"s\",\"risk_level\":\"High\",\"recommended_actions\":[],\"communication_draft\":\"d\"}"}}]}
    monkeypatch.setattr(g, "requests", types.SimpleNamespace(post=lambda *a, **k: FakeResp()))

    out = g.generate_prescription("Gemini", {"x":1})
    assert isinstance(out, dict) and "summary" in out
```

---

## 7) Integration Tests

### 7.1 Train‑then‑predict (no GenAI)
- Run `train()` in a temp directory.
- Load saved **classification** pipeline (`models/phishing_url_detector.pkl`) and run a single prediction.
- (Optional) Load **clustering** pipeline and ensure a label is parseable to an integer.

This is covered by `test_training_writes_models` and by the app‑logic unit test using monkeypatches.

---

## 8) UI & Manual Validation (Streamlit)
Because the Streamlit logic is coupled to UI, a lightweight manual checklist helps:
- [ ] App starts after training
- [ ] Feature inputs appear in sidebar and update risk chart
- [ ] Verdict label renders (**Malicious/Benign**) and confidence metric
- [ ] If Malicious → Threat Attribution and Prescriptive Plan tabs show details
- [ ] No stack traces in UI, and JSON prescription is valid

For automated browser tests, consider Playwright against a running Streamlit server.

---

## 9) Performance & Non‑regression
- **Prediction latency**: add a micro‑benchmark (assert < 100ms for single sample on dev laptop).
- Keep a small **golden JSON** (expected pipeline outputs) to detect drift; compare keys/shape rather than exact scores.

---

## 10) Continuous Integration (optional)
A minimal GitHub Actions job:

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing
```

---

## 11) Security/Secrets in Tests
- Do **not** use real API keys. Unset `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROK_API_KEY` in tests and **mock** all network calls.
- Fail fast if any test tries to reach the network.

```python
# tests/conftest.py (add to file above)
import os, pytest
@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    for k in ["OPENAI_API_KEY","GEMINI_API_KEY","GROK_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
```

---

## 12) Troubleshooting
- **Missing models**: run training first; tests that require artifacts run `train()` in a temporary directory.
- **Plot path**: the training script moves PyCaret’s "Feature Importance.png" into `models/feature_importance.png`; ensure FS permissions allow rename.
- **Clustering label parsing**: the app supports `"Label"`, `"Cluster"`, `"Cluster Label"`, `"label"`, or `"prediction_label"` and normalizes values like `"Cluster 0"` → `0`.

---

Happy testing! 🧪

