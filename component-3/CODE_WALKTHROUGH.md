# Banana Disease Severity & Progression Assessment (Component 3)

## Comprehensive Code Walkthrough & Architecture Guide

Welcome to the **Component 3 Codebase Walkthrough**! This document explains how the entire project is structured, how every module works under the hood, and how data flows through the system—written in clean, intuitive, and simple language.

---

## 1. What Does This Project Do? (The Big Picture)

In agricultural banana farming, diagnosing disease from a leaf photo alone is not enough:

1. **A leaf might have necrotic patches**, but is it a slow-moving leaf spot or a rapid fungal epidemic?
2. **The weather matters enormously**: if it is hot and raining constantly, fungal spores germinate in hours. If it is dry and cold, the disease stops spreading.
3. **Farmers need future projections**: will this plant reach critical damage within 7 days? What specific action or fungicide should be sprayed?

**Component 3 solves this by acting as the multimodal brain:**

- It takes a **leaf image** and a **disease classification** from Component 1.
- It takes **weather transmission urgency** (temperature, humidity, wetness) from Component 2.
- It segments the leaf, measures lesions, computes a **0–100 Severity Score**, projects a **7-Day Progression Forecast**, and generates a **Precision Agronomic Prescription** for Component 4 intervention.

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   Component 1 Vision    │      │  Component 2 Weather    │
│  • Leaf Image Specimen  │      │  • Temperature & RH     │
│  • Pathogen Class & Conf│      │  • Leaf Wetness Hours   │
└───────────┬─────────────┘      └────────────┬────────────┘
            │                                 │
            └───────────────┬─────────────────┘
                            ▼
     ┌───────────────────────────────────────────────┐
     │      Component 3 Multimodal Orchestrator      │
     │                                               │
     │  1. Semantic Segmentation (U-Net 3-Class)     │
     │  2. Morphometrics (Scope %, Lesion Count)     │
     │  3. Biological Scale (Stover-Gauhl Impact)    │
     │  4. Multimodal Fusion Engine (F_hybrid, etc.) │
     │  5. SQLite History & OLS Velocity Analysis    │
     │  6. 7-Day PyTorch GRU Temporal Forecasting    │
     └──────────────────────┬────────────────────────┘
                            ▼
     ┌───────────────────────────────────────────────┐
     │               Actionable Output               │
     │  • Severity Score: 0 - 100 (Low to Critical)  │
     │  • 7-Day Forecast Score & Progression Trend   │
     │  • Precision Agronomic Spray Prescription     │
     └───────────────────────────────────────────────┘
```

---

## 2. Project Directory Map

Here is how the codebase is organized:

```
banana_severity_c3/
│
├── configs/                   # Configuration files
│   └── config.yaml            # Hyperparameters, fusion weights, category thresholds
│
├── data/                      # Local datasets and SQLite storage
│   ├── synthetic/             # Synthetic generated leaf masks & metadata
│   ├── test_images/           # Real Cavendish banana field images
│   └── severity_history.db    # SQLite database storing historical plant visits
│
├── src/                       # Main source code
│   ├── api/                   # FastAPI Web Microservice
│   │   ├── app.py             # REST endpoints (/health, /api/v1/assess-interactive)
│   │   ├── presets.py         # 4 curated demo presets for quick testing
│   │   └── schemas.py         # API request and response data contracts
│   │
│   ├── features/              # Feature engineering
│   │   ├── morphometrics.py   # Lesion scope %, count, coalescence, aspect ratio
│   │   ├── impact_scales.py   # Stover-Gauhl, Cordana & BBTV biological impact
│   │   └── vector_builder.py  # Assembles 8-D feature vector for XGBoost
│   │
│   ├── frontend/              # Interactive Web Dashboard
│   │   ├── index.html         # Tri-modal 3-column UI structure
│   │   ├── style.css          # Modern dark-mode responsive glassmorphic styling
│   │   └── app.js             # Canvas blend viewer, charts, and API interactions
│   │
│   ├── inference/             # Production inference & orchestration
│   │   ├── orchestrator.py    # Master end-to-end inference engine (FR1–FR11)
│   │   ├── database.py        # SQLite history tracking with deduplication
│   │   └── predictors.py      # Runtime model loading & inference wrapper
│   │
│   ├── models/                # Deep learning & ML model architectures
│   │   ├── segmentation/
│   │   │   ├── unet.py        # PyTorch 3-Class U-Net (Background, Lamina, Lesion)
│   │   │   └── classical.py   # OpenCV color-thresholding fallback segmenter
│   │   ├── severity/
│   │   │   ├── fusion_rules.py       # F_mult, F_WS, F_hybrid formulas & thresholds
│   │   │   └── xgboost_regressor.py  # Trained 8-D XGBoost severity regressor
│   │   └── temporal/
│   │       ├── gru_net.py     # PyTorch SeverityGRU recurrent neural network
│   │       └── trend_ols.py   # Ordinary Least Squares velocity regression
│   │
│   ├── schemas/
│   │   └── contracts.py       # Strict Pydantic models for Component 1, 2, and 4
│   │
│   └── training/              # Training scripts for all models
│       ├── seg_trainer.py     # U-Net segmentation training loop
│       ├── xgb_trainer.py     # XGBoost hyperparameter search & training
│       └── gru_trainer.py     # GRU longitudinal sequence training
│
├── tests/                     # 35 automated pytest verification tests
│   ├── test_functional_reqs.py
│   ├── test_phase2_verification.py
│   ├── test_phase3_verification.py
│   ├── test_phase4_verification.py
│   └── test_phase5_verification.py
│
├── results/                   # Trained model checkpoints & evaluation tables
│   ├── unet_weights.pt        # Saved U-Net weights
│   ├── xgboost_model.json     # Saved XGBoost trees
│   └── gru_forecaster.pt      # Saved GRU forecaster weights
│
├── PROJECT_REPORT.md          # Scientific paper & ablation study report
├── implementation_plan.md     # Task-by-task engineering blueprint
└── requirements.txt           # Python library dependencies
```

---

## 3. Step-by-Step Code Walkthrough

Let’s walk through the life of an assessment from the moment an image is submitted to the final 7-day forecast.

---

### Step 1: Strict Data Contracts (`src/schemas/contracts.py`)

To prevent bugs when communicating between Component 1 (vision), Component 2 (weather), and Component 4 (spraying robots), we define strict **Pydantic contracts**:

- **`Component1Input`**: Contains the plant ID, inspection date, diagnosed disease name (e.g., `black_sigatoka`), vision confidence ($0.0–1.0$), and the image path.
- **`Component2Input`**: Contains the weather transmission urgency ($U \in [0.0, 100.0]$), temperature, humidity, and leaf wetness hours.
- **`Component4Output`**: The final packaged result containing the unified severity score ($0–100$), category (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), 7-day forecast score, trend, and the exact agronomic prescription string.

```python
# Example snippet from src/schemas/contracts.py
class Component1Input(BaseModel):
    plant_id: str
    disease_class: DiseaseType  # Enum: black_sigatoka, yellow_sigatoka, cordana, bbtv, healthy
    confidence: float = Field(..., ge=0.0, le=1.0)
    image_path: str
    observation_date: Optional[str] = None
```

**Why this matters:** If an invalid disease name or out-of-range confidence is passed, Pydantic immediately catches it with a clear error message instead of crashing later in the math routines.

---

### Step 2: Semantic Foliar Segmentation (`src/models/segmentation/unet.py`)

When a banana leaf photo comes in, the system needs to know: **where is the leaf, and where is the disease?**

We use a custom **PyTorch 3-Class U-Net**:

- **Class 0 (Background):** Soil, sky, hands, background foliage.
- **Class 1 (Healthy Lamina):** Green, functioning banana leaf tissue.
- **Class 2 (Necrotic Lesion):** Brown, yellow, or black necrotic tissue caused by fungal streaks or virus symptoms.

#### How U-Net Works (Simply Explained):

1. **Contracting Path (Encoder):** The image is repeatedly downsampled with `MaxPool2d` and `DoubleConv` blocks. This captures high-level context (*"This is a banana leaf with spots"*).
2. **Expansive Path (Decoder):** The image is upsampled back to its original $256 \times 256$ resolution using transposed convolutions.
3. **Skip Connections:** High-resolution spatial information from the encoder is concatenated directly to the decoder. This ensures the model draws razor-sharp borders around tiny early fungal streaks.

```python
# Core forward pass in src/models/segmentation/unet.py
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x1 = self.inc(x)        # Initial double conv
    x2 = self.down1(x1)     # Downsample 1
    x3 = self.down2(x2)     # Downsample 2
    x4 = self.down3(x3)     # Downsample 3
    x5 = self.down4(x4)     # Bottleneck
  
    x = self.up1(x5, x4)    # Upsample with skip connection from x4
    x = self.up2(x, x3)     # Upsample with skip connection from x3
    x = self.up3(x, x2)     # Upsample with skip connection from x2
    x = self.up4(x, x1)     # Upsample with skip connection from x1
    logits = self.outc(x)   # Final 1x1 conv yielding 3 class channels
    return logits
```

**Graceful Degradation (`src/models/segmentation/classical.py`):**
If the server is ever running in a lightweight environment without GPU or PyTorch weights, it seamlessly falls back to `ClassicalSegmenter`, which uses HSV color space thresholds and morphological filtering to segment the leaf and lesions.

---

### Step 3: Morphometric Feature Extraction (`src/features/morphometrics.py`)

A raw segmented mask is just an array of pixels (`0`, `1`, `2`). We need to extract meaningful biological metrics. The function `extract_leaf_and_lesion_features()` converts the mask into a `MorphometricFeatures` dataclass:

1. **Foliar Scope % ($S$):** The percentage of the leaf destroyed by lesions:
   $$
   S = \frac{\text{Lesion Pixels}}{\text{Healthy Pixels} + \text{Lesion Pixels}} \times 100
   $$
2. **Lesion Count:** Uses OpenCV `cv2.connectedComponentsWithStats` to count distinct disease spots, filtering out pixel noise ($< 5\text{ px}$).
3. **Coalescence Ratio % ($R_{\text{coal}}$):** Measures whether small spots have merged into large continuous dead patches by comparing the largest connected lesion against the total leaf area.
4. **Leaf Aspect Ratio:** The length-to-width ratio of the leaf contour. A healthy Cavendish leaf has an aspect ratio around $3.5$.
5. **Lamina Distortion Index ($D_{\text{morph}}$):** Detects stunting or "bunchy top" deformation when the aspect ratio compresses from $3.5$ down toward $1.5$.

---

### Step 4: Biological Pathogen Impact Scales (`src/features/impact_scales.py`)

**Key Insight:** $10\%$ leaf damage from **Black Sigatoka** is far more dangerous than $10\%$ damage from a minor leaf spot like **Cordana**, because Sigatoka destroys photosynthetic capacity and causes premature fruit ripening.

The function `calculate_disease_impact()` converts the visual scope % into a standardized **Biological Impact Score ($I \in [0.0, 100.0]$)**:

- **Sigatoka (Black & Yellow):** Implements the internationally recognized **Stover-Gauhl International Scale (Grades 0 to 6)** using piecewise-linear interpolation across 7 validated boundaries:
  - Grade 0: $0\%$ (Healthy)
  - Grade 1: $< 5\%$ isolated flecks (Impact $\approx 5$)
  - Grade 2: $5\% - 15\%$ spots (Impact $\approx 13$)
  - Grade 3: $15\% - 35\%$ spots (Impact $\approx 23$)
  - Grade 4: $35\% - 50\%$ spots (Impact $\approx 40$)
  - Grade 5: $50\% - 75\%$ severe dead leaf halves (Impact $\approx 65$)
  - Grade 6: $> 75\%$ complete foliar collapse (Impact $\approx 100$)
- **Cordana:** Uses Horsfall-Barratt logarithmic scaling combined with lesion coalescence.
- **BBTV (Bunchy Top):** Uses morphological distortion and leaf stunting proxies.
- **Healthy:** Fixed at $I = 0.0$.

---

### Step 5: The Four Multimodal Fusion Formulations (`src/models/severity/fusion_rules.py`)

Now we have:

- $I$: Biological Pathogen Impact ($0–100$)
- $S$: Foliar Necrotic Scope ($0–100\%$)
- $U$: Weather Transmission Urgency ($0–100\%$)

How do we fuse them into a single definitive Severity Score? We implemented and benchmarked 4 formulas:

#### 1. Multiplicative Baseline ($F_{\text{mult}}$)

$$
F_{\text{mult}} = \frac{I \times U \times S}{10,000}
$$

- **Why it fails (The "Zero-Collapse" Bug):** If the weather is dry ($U = 0$), the entire formula multiplies to zero ($0.0/100$)! A banana leaf with 80% dead tissue would be reported as completely healthy—a lethal false negative in plant pathology.

#### 2. Weighted-Sum Late Fusion ($F_{\text{WS}}$)

$$
F_{\text{WS}} = 0.45 \cdot I + 0.35 \cdot S + 0.20 \cdot U
$$

- **Limitation:** Prevents zero-collapse, but treats weather as a simple additive offset instead of an environmental amplifier.

#### 3. Damped Multiplicative Hybrid ($F_{\text{hybrid}}$ — Recommended Operational Model) ⭐

$$
F_{\text{hybrid}} = (0.55 \cdot I + 0.45 \cdot S) \times \left(0.70 + 0.30 \cdot \frac{U}{100}\right)
$$

- **Why this works brilliantly:**
  - **Biological Baseline Floor ($0.55I + 0.45S$):** Evaluates actual physical leaf damage.
  - **Damped Environmental Modulator ($0.70 + 0.30 \times \frac{U}{100}$):** Bounded between $0.70$ and $1.00$.
  - In dry weather ($U = 0\%$), the score preserves **at least 70%** of the biological necrosis, completely eliminating the Zero-Collapse flaw.
  - In wet weather ($U = 100\%$), it scales up to full 100% severity.
  - Achieves the lowest test error ($\text{RMSE} = 3.9160$, $\text{MAE} = 2.9433$, Spearman $\rho = 0.9724$).

#### 4. Learned Gradient Boosting ($F_{\text{XGB}}$ — Machine Learning Model)

Trained with `xgboost.XGBRegressor` on an 8-dimensional feature vector ($I, S, U, \text{num\_lesions}, R_{\text{coal}}, \text{aspect\_ratio}, \text{disease\_id}, \text{confidence}$). Validates that biological impact ($69.5\%$) and scope ($12.8\%$) dominate real severity.

#### Severity Categorization (FR7):

Scores are mapped into validated agricultural action bands:

- **`LOW` ($0.0 \le \text{Score} < 20.0$):** Routine monitoring; biological control.
- **`MODERATE` ($20.0 \le \text{Score} < 50.0$):** Preventive bio-fungicide; de-leafing necrotic tips.
- **`HIGH` ($50.0 \le \text{Score} < 75.0$):** Targeted systemic fungicide; sanitation pruning.
- **`CRITICAL` ($75.0 \le \text{Score} \le 100.0$):** Emergency containment quarantine; systemic eradicate spray.

---

### Step 6: Longitudinal History Tracking (`src/inference/database.py`)

A single photo shows a snapshot in time. To know if a plant is getting better or worse, we store historical observations in a local SQLite database (`data/severity_history.db`).

- **Deduplication / Upsert:** Uses `INSERT ... ON CONFLICT(plant_id, observation_date) DO UPDATE` to ensure multiple assessments on the same date update the existing record rather than creating duplicate overlapping points on timeline graphs.
- **Querying History:** `get_plant_history(plant_id)` returns chronological observations sorted by date.

---

### Step 7: Temporal Forecasting & Trend Analysis (`src/models/temporal/`)

With historical visits available, we run two complementary temporal analyses:

#### 1. Velocity & Trend Classification (`trend_ols.py`)

Computes an Ordinary Least Squares (OLS) linear slope over the observation dates:

$$
\text{Slope} = \frac{\Delta \text{Severity}}{\Delta \text{Days}} \quad (\%/\text{day})
$$

- $\text{Slope} > +1.5\%/\text{day} \implies$ **Rapidly Deteriorating**
- $+0.2\% < \text{Slope} \le +1.5\%/\text{day} \implies$ **Deteriorating**
- $-0.5\% \le \text{Slope} \le +0.2\%/\text{day} \implies$ **Stable**
- $\text{Slope} < -0.5\%/\text{day} \implies$ **Improving**

#### 2. PyTorch GRU 7-Day Severity Forecaster (`gru_net.py`)

Uses a Gated Recurrent Unit (GRU) neural network (`SeverityGRU`) to project severity 7 days into the future:

- **Input sequence:** Historical vectors of $[s_t, u_t, \text{scope}_t, i_t, \text{disease\_id}_t]$.
- **Sequence Masking:** Uses valid length masking to prevent trailing zero-padding from corrupting the hidden states.
- **Output:** Predicted future severity score ($0.0–100.0$).

---

### Step 8: Master Orchestrator (`src/inference/orchestrator.py`)

The `Component3Orchestrator` is the central coordinator that brings all the steps together:

```python
# How orchestrator.py runs the full pipeline
orch = Component3Orchestrator()
output = orch.assess_and_forecast(
    c1_input=Component1Input(...),
    c2_input=Component2Input(...),
    fusion_method="F_hybrid"
)
```

**What it executes internally:**

1. Loads and preprocesses the leaf image ($256 \times 256$ RGB tensor).
2. Runs U-Net to produce the 3-class segmentation mask.
3. Computes morphometrics ($S$, lesion count, coalescence).
4. Calculates biological impact ($I$) using Stover-Gauhl.
5. Fuses modalities using the selected strategy ($F_{\text{hybrid}}$).
6. Assigns the severity category and generates the agronomic prescription.
7. Stores the observation in SQLite (`severity_history.db`).
8. Computes OLS trend slope and runs GRU 7-day forecasting.
9. Packages everything into `Component4Output`.

---

### Step 9: Web API Microservice (`src/api/app.py` & `src/api/presets.py`)

Built with **FastAPI** for high performance and automatic OpenAPI documentation:

- **`GET /health`**: Verifies that U-Net, XGBoost, and GRU models are loaded in memory.
- **`GET /api/v1/demo-presets`**: Returns 4 curated field scenarios:
  1. *Early Yellow Sigatoka* (Grade 1 flecks, Moderate risk)
  2. *Severe Black Sigatoka Outbreak* (Grade 5 necrosis, High urgency)
  3. *Cordana Leaf Spot* (Large coalescing concentric patches)
  4. *Healthy Cavendish Control* (Grade 0, Baseline check)
- **`POST /api/v1/assess-interactive`**: The primary assessment endpoint. Accepts base64 images or preset paths and returns the full assessment, base64 segmentation overlay, timeline history, and future projections.

---

### Step 10: Interactive Web Dashboard (`src/frontend/`)

A responsive, high-aesthetic web interface served directly from the FastAPI root (`http://127.0.0.1:8000/`):

- **Column 1 (Input Controls):** Plant ID, inspection date picker, disease selector, image dropzone, interactive weather urgency slider, and fusion strategy selector pills.
- **Column 2 (Semantic Segmentation):** $256 \times 256$ Canvas viewport with a real-time mask opacity blend slider ($0\%$ raw leaf to $100\%$ mask overlay) and the 4 morphometrics cards.
- **Column 3 (Decision & Forecasting):** Animated circular SVG severity score ring, 7-day GRU forecast metric card, actionable intervention prescription box, and the HTML5 longitudinal progression timeline chart.
- **Bottom Panel (Zero-Collapse Stress Test Simulator):** An interactive educational lab demonstrating how $F_{\text{mult}}$ collapses to 0 when weather urgency $U=0$, while $F_{\text{hybrid}}$ maintains a safe $70\%$ biological floor.

---

## 4. How to Run, Test, and Train Everything

### 1. Launch the Application

Start the FastAPI server:

```bash
# Using the project virtual environment
.venv/Scripts/python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to: **`http://127.0.0.1:8000/`** (Interactive Dashboard)
API documentation: **`http://127.0.0.1:8000/docs`** (Swagger UI)

### 2. Run the Automated Test Suite

To verify all 35 functional requirements, regression tests, and mathematical contracts:

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

### 3. Retrain Models

To retrain any model from scratch using custom or synthetic data:

- **Train U-Net:**
  ```bash
  .venv/Scripts/python.exe src/segmentation/train_seg.py --epochs 10 --batch-size 16
  ```
- **Train XGBoost Regressor:**
  ```bash
  .venv/Scripts/python.exe -m src.training.xgb_trainer
  ```
- **Train PyTorch GRU Forecaster:**
  ```bash
  .venv/Scripts/python.exe -m src.training.gru_trainer
  ```

---

## 5. Summary: Key Engineering Takeaways

1. **True Multimodality:** Combines computer vision (pixels), agronomic biology (pathogen scales), and IoT microclimate (weather sensors).
2. **Mathematically Bulletproof:** Solved the literature "Zero-Collapse" flaw using the Damped Multiplicative Hybrid ($F_{\text{hybrid}}$).
3. **Longitudinal Intelligence:** Doesn't just classify a single image; tracks disease velocity over time and uses recurrent neural networks (GRU) to forecast 7 days ahead.
4. **Resilient Architecture:** Features graceful fallbacks (Classical segmentation if deep weights are missing; OLS trend if historical sequences are short).
5. **Production-Ready:** Fully typed with Pydantic contracts, covered by 35 unit tests, and accessible via a modern web interface.
