# Component 3: Architectural Specification & Mathematical Standards
**Multimodal Disease Severity Assessment and Temporal Progression Forecasting for Cavendish Banana Plantations**

---

## 1. Scope & System Boundary

Component 3 receives diagnosis results from **Component 1** (`disease_label`, `disease_confidence`, `image_path`) and environmental transmission urgency from **Component 2** (`urgency_score`, `zone_id`), assesses multidimensional physical foliar severity, persists longitudinal temporal records, estimates 7-day progression trends, and produces validated decision outputs for **Component 4** (Precision Intervention Engine).

```
   [Component 1: Vision Diagnosis]     [Component 2: Climate & Microclimate]
               │                                         │
               │ (Component1Input)                       │ (Component2Input)
               ▼                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       COMPONENT 3 ARCHITECTURE                          │
   │                                                                         │
   │  1. Semantic Segmentation (U-Net 3-Class) -> [Leaf Mask, Lesion Mask]   │
   │  2. Morphometric Scope Engine -> Scope %, Lesion Count, Coalescence, AR │
   │  3. Disease-Specific Impact Engine -> Impact Score I ∈ [0, 100]         │
   │  4. Multimodal Fusion Engine -> F_mult, F_WS, F_hybrid, F_XGB           │
   │  5. Severity Categorization -> Low, Moderate, High, Critical            │
   │  6. Temporal Database Store (SQLite) -> Longitudinal Tracking           │
   │  7. Trend Classifier (OLS β_7d, ±3.0) & 7-Day GRU Forecaster            │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │ (Component4Output)
                                        ▼
                       [Component 4: Precision Action Engine]
```

---

## 2. Supported Disease Taxonomy

The system supports exactly 5 target classes:
1. `black_sigatoka` (*Pseudocercospora fijiensis*)
2. `yellow_sigatoka` (*Pseudocercospora musae*)
3. `cordana` (*Neocordana musae*)
4. `bbtv` (*Banana bunchy top virus*)
5. `healthy` (Asymptomatic Cavendish leaf)

---

## 3. Mathematical Formulations & Impact Scales

### 3.1 Scope Percentage (Foliar Coverage)
$$\text{Scope } \% = \frac{\text{Lesion Pixels}}{\text{Leaf Pixels}} \times 100.0$$
- If $\text{Leaf Pixels} == 0$, $\text{Scope } \% = 0.0$.
- Bounded strictly within $[0.0, 100.0]$.

### 3.2 Disease-Specific Impact Calculators ($I \in [0.0, 100.0]$)

#### A. Black & Yellow Sigatoka (Stover-Gauhl Scale & Pinzon-Nunez Bands)
Bands for Scope $S$:
- Grade 0: $S = 0.0\%$ $\implies I = 0.0$
- Grade 1: $0.0\% < S \le 5.0\%$ $\implies I = 0.0 + 16.67 \times \left(\frac{S - 0.0}{5.0}\right)$
- Grade 2: $5.0\% < S \le 13.0\%$ $\implies I = 16.67 + 16.67 \times \left(\frac{S - 5.0}{8.0}\right)$
- Grade 3: $13.0\% < S \le 23.0\%$ $\implies I = 33.33 + 16.67 \times \left(\frac{S - 13.0}{10.0}\right)$
- Grade 4: $23.0\% < S \le 40.0\%$ $\implies I = 50.00 + 16.67 \times \left(\frac{S - 23.0}{17.0}\right)$
- Grade 5: $40.0\% < S \le 65.0\%$ $\implies I = 66.67 + 16.67 \times \left(\frac{S - 40.0}{25.0}\right)$
- Grade 6: $65.0\% < S \le 100.0\%$ $\implies I = 83.33 + 16.67 \times \left(\frac{S - 65.0}{35.0}\right)$

Multi-leaf Plant Disease Severity Index (DSI) for $T$ leaves:
$$\text{DSI} = \frac{\sum_{b=0}^{6} (n_b \times b)}{6 \times T} \times 100.0$$

#### B. Cordana Leaf Spot (Morphometric Coalescence Proxy)
- Base foliar score $I_{\text{HB}}(S)$ mapped across Horsfall-Barratt bands:
  $S=0 \to 0$, $(0, 5] \to 20$, $(5, 15] \to 40$, $(15, 30] \to 60$, $(30, 50] \to 80$, $>50 \to 100$.
- Coalescence Ratio: $R_{\text{coal}} = (\text{Area of largest single lesion} / \text{Leaf Pixels}) \times 100.0$.
- Lesion Count Score: $D_{\text{count}} = \min(100.0, (\text{Lesion Count} / 20.0) \times 100.0)$.
- Composite Impact:
  $$I_{\text{Cordana}} = \text{clip}\left(0.65 \cdot I_{\text{HB}}(S) + 0.20 \cdot \min(100.0, 2.5 \cdot R_{\text{coal}}) + 0.15 \cdot D_{\text{count}},\; 0.0,\; 100.0\right)$$

#### C. Banana Bunchy Top Virus (BBTV Symptom-Stage Proxy)
- Reference healthy aspect ratio $\text{AR}_{\text{ref}} = 3.5$, minimum stunted ratio $= 1.5$.
- Lamina Distortion:
  $$D_{\text{morph}} = \text{clip}\left(\frac{3.5 - \text{AR}_{\text{leaf}}}{3.5 - 1.5} \times 100.0,\; 0.0,\; 100.0\right)$$
- Continuous Impact:
  - If $S < 1.0\%$ and $D_{\text{morph}} < 10.0 \implies I = 0.0$ (Healthy control)
  - Else:
    $$I_{\text{BBTV}} = \text{clip}\left(25.0 + 0.45 \cdot \min(100.0, 2.0 \cdot S) + 0.30 \cdot D_{\text{morph}},\; 0.0,\; 100.0\right)$$

#### D. Healthy Control
- $I_{\text{Healthy}} = 0.0$, Stover-Gauhl Grade $= 0$.

---

## 4. Multimodal Fusion Strategies

Four benchmarking strategies combine Biological Impact ($I$), Foliar Scope ($S$), and Environmental Transmission Urgency ($U \in [0, 100]$):

1. **Multiplicative Baseline ($F_{\text{mult}}$):**
   $$F_{\text{mult}} = \frac{I \times U \times S}{10,000.0}$$
   *(Limitation: Collapses to zero if $U \to 0$ regardless of leaf necrosis).*

2. **Weighted-Sum Late Fusion ($F_{\text{WS}}$):**
   $$F_{\text{WS}} = 0.45 \cdot I + 0.35 \cdot S + 0.20 \cdot U$$

3. **Damped Multiplicative Hybrid ($F_{\text{hybrid}}$ — Recommended Operational):**
   $$F_{\text{hybrid}} = \underbrace{(0.55 \cdot I + 0.45 \cdot S)}_{\text{Biological Foliar Severity}} \times \underbrace{\left(0.70 + 0.30 \cdot \frac{U}{100.0}\right)}_{\text{Environmental Urgency Amplifier}}$$

4. **Learned XGBoost Fusion ($F_{\text{XGB}}$):**
   8-dimensional feature vector:
   $$\mathbf{x} = [I,\; S,\; U,\; \text{disease\_id},\; N_{\text{lesions}},\; R_{\text{coal}},\; D_{\text{morph}},\; \text{AR}_{\text{leaf}}]$$

---

## 5. Severity Categorization Thresholds

| Severity Category | Composite Severity Score Range | Equivalent Foliar Scope ($S$) | Recommended Intervention |
|:---|:---:|:---:|:---|
| **Low** | $[0.00, 20.00)$ | $0\% \text{ -- } 5\%$ | Routine scheduled scouting; no chemical intervention. |
| **Moderate** | $[20.00, 50.00)$ | $5\% \text{ -- } 15\%$ | Targeted surgical de-leafing of necrotic tips; preventive bio-fungicide. |
| **High** | $[50.00, 75.00)$ | $15\% \text{ -- } 40\%$ | Priority systemic fungicide spray & drainage check. *(Validation: 72.0 maps to High)* |
| **Critical** | $[75.00, 100.00]$ | $> 40\%$ or BBTV Stage 3-4 | Immediate quarantine, whole-leaf pruning, or mat eradication (for BBTV). |

---

## 6. Temporal Progression & 7-Day Forecasting

### 6.1 Weekly Rate of Change ($\beta_{\text{7d}}$)
For $n$ chronological observations $(t_1, Y_1), \dots, (t_n, Y_n)$ where $t$ is in days:
$$\beta_{\text{7d}} = 7.0 \times \frac{\sum_{i=1}^{n} (t_i - \bar{t})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (t_i - \bar{t})^2}$$

### 6.2 Progression Trend Thresholds ($\epsilon = \pm 3.0$ points/7d)
- **`Worsening`:** $\beta_{\text{7d}} > +3.0$
- **`Stable`:** $-3.0 \le \beta_{\text{7d}} \le +3.0$
- **`Improving`:** $\beta_{\text{7d}} < -3.0$

### 6.3 Cold-Start Reliability Protocol
- **$n = 1$:** `trend = INITIAL_OBSERVATION`, `forecast_7d = None`, `confidence_flag = "LOW_COLD_START"`.
- **$n = 2$:** Two-point linear extrapolation $\hat{Y}_{t+7} = \text{clip}(Y_2 + \beta_{\text{7d}}, 0.0, 100.0)$, `confidence_flag = "MEDIUM"`.
- **$n \ge 3$:** Full OLS + PyTorch GRU 7-day inference, `confidence_flag = "HIGH"`.
