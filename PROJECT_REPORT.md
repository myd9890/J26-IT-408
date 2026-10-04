# 📋 Project Report — Component 3

## Multimodal Disease Severity Assessment & Temporal Progression Forecasting for Cavendish Banana Plantations

**Author:** Oshada Kavinda
**Date:** October 18, 2026
**Status:** ✅ All 38 tasks completed | 35/35 tests passing

---

## 1. What This Project Does (Simple Explanation)

This project is **Component 3** of a larger banana disease detection system. It sits between:

- **Component 1** (takes a photo of a banana leaf and tells you what disease it has)
- **Component 2** (looks at weather/climate data and gives a risk urgency score)
- **Component 4** (takes the severity output and plans what to do — spray, prune, quarantine, etc.)

**Component 3's job is to answer**: *"How bad is the disease, and will it get worse next week?"*

It does this in **7 steps**:

1. **Segment the leaf image** — uses a U-Net neural network to separate the leaf from the background, and find exactly where the disease spots are on the leaf.
2. **Measure the damage** — calculates what percentage of the leaf is diseased (scope %), counts lesion spots, measures how merged they are, and checks if the leaf shape is abnormally stunted.
3. **Score the disease impact** — uses different scoring scales depending on the disease type (Sigatoka gets a Stover-Gauhl grade, Cordana uses a coalescence formula, BBTV checks stunting).
4. **Fuse everything together** — combines the disease score, leaf damage percentage, and weather urgency into one final severity number (0–100).
5. **Categorize** — maps the score to Low / Moderate / High / Critical.
6. **Track over time** — saves each inspection to a SQLite database so it can track how each plant is doing over weeks.
7. **Forecast** — uses a GRU neural network to predict what the severity will be in 7 days.

---

## 2. Diseases Supported

| Disease                     | Scientific Name                | How It's Scored                                     |
| :-------------------------- | :----------------------------- | :-------------------------------------------------- |
| **Black Sigatoka**    | *Pseudocercospora fijiensis* | 7-band Stover-Gauhl scale (Grades 0–6)             |
| **Yellow Sigatoka**   | *Pseudocercospora musae*     | Same Stover-Gauhl scale as Black Sigatoka           |
| **Cordana Leaf Spot** | *Neocordana musae*           | Horsfall-Barratt bands + lesion coalescence + count |
| **BBTV**              | *Banana bunchy top virus*    | Symptom stage (0–4) + leaf aspect ratio distortion |
| **Healthy**           | —                             | Always scores 0 (baseline control)                  |

---

## 3. Project Structure

```
banana_severity_c3/
├── configs/
│   └── config.yaml                    # All hyperparameters, thresholds, weights
├── src/
│   ├── schemas/contracts.py           # Pydantic data contracts (input/output models)
│   ├── data/
│   │   ├── public_adapter.py          # Maps real dataset folder names to disease classes
│   │   ├── synthetic_generator.py     # Creates fake training data (120 plants × 6 weeks)
│   │   └── dataset.py                 # Train/val/test splitting + PyTorch Dataset
│   ├── segmentation/
│   │   ├── classical_baseline.py      # ExG + Otsu + CIELAB rule-based segmenter
│   │   ├── unet.py                    # 3-class U-Net neural network
│   │   ├── loss_and_metrics.py        # Dice + CrossEntropy loss, IoU/Dice metrics
│   │   ├── scope_calculator.py        # Extracts scope %, coalescence, aspect ratio
│   │   └── train_seg.py              # Training loop for U-Net
│   ├── severity/
│   │   ├── impact_calculators.py      # Stover-Gauhl, Cordana, BBTV scoring formulas
│   │   ├── fusion_strategies.py       # F_mult, F_WS, F_hybrid formulas + categorization
│   │   └── xgboost_engine.py          # 8-feature XGBoost regressor + classifier
│   ├── temporal/
│   │   ├── database.py                # SQLite store for plant observation history
│   │   ├── trend_classifier.py        # OLS weekly slope + Worsening/Stable/Improving
│   │   └── gru_forecaster.py          # PyTorch GRU 7-day severity predictor
│   ├── pipeline/
│   │   └── c3_orchestrator.py         # Master pipeline connecting everything
│   ├── api/
│   │   └── app.py                     # FastAPI REST endpoints
│   └── evaluation/
│       ├── expert_simulator.py        # Simulates human expert ratings with noise
│       └── run_all_benchmarks.py      # Generates all thesis tables + figures
├── tests/                             # 5 test files, 35 test cases total
├── results/
│   ├── unet_weights.pt               # Trained U-Net (23.4 MB)
│   ├── xgboost_model.json            # Trained XGBoost severity model (711 KB)
│   ├── gru_forecaster.pt             # Trained GRU forecaster (20.9 KB)
│   ├── tables/                        # 4 CSV benchmark tables
│   └── figures/                       # 4 PNG publication figures (300 DPI)
├── data/
│   ├── raw/                           # Original downloaded datasets
│   ├── processed/                     # 3,985 real images (5 disease classes @ 256×256)
│   └── synthetic/                     # 720 generated records with images + masks
└── requirements.txt                   # Python dependencies
```

**Total: 23 source files + 5 test files + 3 trained models + 4 tables + 4 figures**

---

## 4. Technologies Used

| Category                   | Technologies                                                   |
| :------------------------- | :------------------------------------------------------------- |
| **Deep Learning**    | PyTorch 2.0+ (U-Net segmentation, GRU forecaster)              |
| **Machine Learning** | XGBoost, Scikit-Learn (Random Forest, SVR baselines)           |
| **Computer Vision**  | OpenCV (morphometrics, contour analysis, connected components) |
| **API**              | FastAPI + Uvicorn (REST microservice)                          |
| **Data Validation**  | Pydantic v2 (strict schema contracts)                          |
| **Database**         | SQLite (temporal observation store)                            |
| **Visualization**    | Matplotlib, Seaborn (publication-grade 300 DPI figures)        |
| **Testing**          | Pytest (35 tests across 5 files)                               |

---

## 5. How the Fusion Formulas Work

The system combines 3 numbers into one severity score:

- **I** = Disease impact (biological damage, 0–100)
- **S** = Scope % (how much of the leaf is affected)
- **U** = Urgency (weather risk from Component 2, 0–100)

Four fusion strategies are implemented:

| Strategy                                  | Formula                                         | Problem                                                                 |
| :---------------------------------------- | :---------------------------------------------- | :---------------------------------------------------------------------- |
| **Multiplicative**                  | `(I × U × S) / 10000`                       | ❌ Collapses to 0 when weather is calm (U=0), even if the leaf is dying |
| **Weighted-Sum**                    | `0.45×I + 0.35×S + 0.20×U`                 | ⚠️ Linear, doesn't capture interactions                               |
| **Damped Hybrid** *(recommended)* | `(0.55×I + 0.45×S) × (0.70 + 0.30×U/100)` | ✅ Keeps at least 70% of bio severity even with zero urgency            |
| **XGBoost**                         | Learned from 8 features                         | ✅ Best accuracy, trained non-linear model                              |

---

## 6. Benchmark Results Summary

### Segmentation (U-Net vs Classical)

| Model                           |    Lesion IoU    |   Lesion Dice   |    Macro Dice    |
| :------------------------------ | :--------------: | :--------------: | :--------------: |
| Classical Baseline (ExG + Otsu) |      0.8809      |      0.9367      |      0.9581      |
| **PyTorch U-Net (Ours)**  | **0.9103** | **0.9530** | **0.9764** |

### Severity Estimation (XGBoost vs baselines)

| Model                    |       MAE       |      RMSE      |       R²       |
| :----------------------- | :-------------: | :-------------: | :-------------: |
| Rule-Based (Hybrid)      |      3.12      |      4.35      |      0.961      |
| Random Forest            |      1.45      |      2.11      |      0.991      |
| **XGBoost (Ours)** | **0.845** | **1.218** | **0.998** |

### 7-Day Forecasting (GRU vs baselines)

| Model                           |    7-Day MAE    |   7-Day RMSE   |
| :------------------------------ | :-------------: | :-------------: |
| OLS Extrapolation               |      6.84      |      9.42      |
| **GRU Forecaster (Ours)** | **4.538** | **7.355** |

---

## 7. Test Results

```
✅ 35/35 tests passed (42.71 seconds)
   ├── test_functional_reqs.py       — 11 tests (FR1 through FR11)
   ├── test_phase2_verification.py   —  5 tests (data pipeline)
   ├── test_phase3_verification.py   —  7 tests (segmentation + morphometrics)
   ├── test_phase4_verification.py   —  8 tests (impact, fusion, XGBoost)
   └── test_phase5_verification.py   —  4 tests (database, trends, GRU)
```

**1 warning:** Starlette deprecation for `httpx` vs `httpx2` in test client (non-blocking).

---

## 8. API Endpoints

| Method   | Endpoint                             | Description                                                  |
| :------- | :----------------------------------- | :----------------------------------------------------------- |
| `POST` | `/api/v1/assess-and-forecast`      | Full pipeline: image → segmentation → severity → forecast |
| `GET`  | `/api/v1/plant-history/{plant_id}` | Retrieves chronological inspection timeline for a plant      |
| `GET`  | `/health`                          | Service healthcheck + model loading status                   |

---

## 9. 🐛 Bugs & Code Issues Found

### Bug 1: Duplicate `forward()` Method in U-Net

**File:** `src/segmentation/unet.py` (Lines 138–167)

The `UNet` class has **two `forward()` methods**. The first one (lines 138–153) contains broken/experimental code with an invalid walrus operator expression:

```python
x4 = self.down3(x2 := x2, x3)[1] if False else self.down2(x2)  # clear ref
```

Python silently uses the **second** `forward()` definition (line 155), so it works at runtime, but this is messy and confusing. The dead code should be removed.

**Severity:** Low (works correctly because Python uses the last definition, but unprofessional)

### Bug 2: CORS Wildcard in API

**File:** `src/api/app.py` (Lines 29–35)

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

This allows any website to call the API, which is a security risk in production.

**Severity:** Low for academic project, but would be a real issue in deployment.

### Bug 3: Hardcoded Absolute Path in Synthetic Generator

**File:** `src/data/synthetic_generator.py` (Line 28)

```python
SYNTHETIC_DIR = r"d:\assignments\banana\banana_severity_c3\data\synthetic"
```

This hardcoded Windows path will break on any other machine. Should be relative.

**Severity:** Medium — anyone cloning this repo will need to change this manually.

### Bug 4: Starlette Deprecation Warning

The test client uses the deprecated `httpx` integration instead of `httpx2`:

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated
```

**Severity:** Low — will break in a future FastAPI/Starlette version.

---

## 10. 📝 Things Still To Do

### Should-Do (Improvements)

| #  | Task                                              | Priority | Details                                                                                            |
| :- | :------------------------------------------------ | :-------: | :------------------------------------------------------------------------------------------------- |
| 5  | **Add GPU acceleration support**            | 🟡 Medium | U-Net inference defaults to CPU even when GPU is available in some code paths                      |
| 6  | **Add logging instead of print statements** | 🟡 Medium | Many files use `print()` for status — should use Python `logging` module                      |
| 7  | **Add input image validation**              | 🟡 Medium | API accepts `image_path` but has no validation for corrupt/missing images before pipeline starts |
| 8  | **Add rate limiting to API**                |  🟢 Low  | No rate limiting on the FastAPI endpoints                                                          |
| 9  | **Add Dockerization**                       |  🟢 Low  | No `Dockerfile` or `docker-compose.yml` for containerized deployment                           |
| 10 | **Add CI/CD pipeline config**               |  🟢 Low  | No GitHub Actions or similar CI config for automated testing                                       |

### Nice-To-Have (Future Work)

| #  | Task                                | Details                                                                                                                                    |
| :- | :---------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| 11 | **Real-world validation**     | All benchmarks use synthetic data + simulated expert ground truth — needs validation on real field photos with actual agronomist ratings  |
| 12 | **Multi-leaf DSI endpoint**   | The `calculate_stover_gauhl_dsi()` function exists but isn't exposed in the API — add an endpoint for whole-plant multi-leaf assessment |
| 13 | **Model retraining pipeline** | No automated way to retrain U-Net/XGBoost/GRU on new data — currently requires manual script execution                                    |
| 14 | **Add Prometheus metrics**    | For monitoring API latency, throughput, and error rates in production                                                                      |
| 15 | **WebSocket streaming**       | For real-time field monitoring of disease progression                                                                                      |

---

## 11. ⚠️ Potential Improvements

### Code Quality

- **The U-Net uses `base_filters=16`** in training (small model), while the architecture docstring says 64/128/256/512/1024 channels. This is intentional for speed on synthetic data but would need scaling up for real deployment.
- **XGBoost feature names** aren't passed during training, which may cause warnings in newer XGBoost versions.
- **The `BananaSegmentationDataset`** class supports augmentations but they aren't configurable from `config.yaml`.

### Architecture

- **SQLite** is fine for a prototype but won't scale to concurrent multi-user access in production. Consider PostgreSQL or a cloud database.
- **The orchestrator** loads all 3 models (U-Net, XGBoost, GRU) into memory on startup, even if only one fusion method is needed. Lazy loading could reduce memory usage.
- **The API uses a global singleton pattern** for the orchestrator — this works for single-server deployment but won't work for multi-worker setups without shared state.

### Data

- **Training data is 100% synthetic** (720 generated records). While 3,985 real images exist in `data/processed/`, the pipeline trains on synthetic data. The real images are only used for pseudo-mask generation through the public adapter.
- **The expert simulator** uses Gaussian noise (σ=4.2%) which is a simplification. Real inter-rater variability is non-Gaussian and disease-dependent.

---

## 12. Summary

| Metric                        | Value                                                       |
| :---------------------------- | :---------------------------------------------------------- |
| **Lines of Code**       | ~4,500+ lines of Python                                     |
| **Source Files**        | 23 modules across 8 packages                                |
| **Test Files**          | 5 files, 35 test cases                                      |
| **Trained Models**      | 3 (U-Net 23.4MB, XGBoost 711KB, GRU 20.9KB)                 |
| **Benchmark Tables**    | 4 CSV files                                                 |
| **Publication Figures** | 4 PNG files @ 300 DPI                                       |
| **Dataset**             | 3,985 real + 720 synthetic images                           |
| **Diseases Covered**    | 5 (Black Sigatoka, Yellow Sigatoka, Cordana, BBTV, Healthy) |
| **Test Pass Rate**      | 100% (35/35)                                                |
| **Task Completion**     | 100% (38/38 tasks)                                          |

> **Bottom line:** The project is functionally complete and all tests pass. The main issues are the duplicate `forward()` method in the U-Net (dead code that should be cleaned up), a hardcoded absolute path in the synthetic generator, and the fact that all benchmarks run on synthetic data rather than real-world validation.
