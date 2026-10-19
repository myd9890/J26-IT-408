/**
 * Banana Disease Multimodal Severity & Progression Demo (Component 3)
 * ES6 Client Controller
 */

// Application State
const state = {
  presets: [],
  activePreset: null,
  customImageBase64: null,
  currentAssessment: null,
  currentHistory: null,
  currentImpactScore: 50.0,
  maskOpacity: 0.85,
  selectedFusionStrategy: 'F_hybrid',
  rawImageObj: null,
  maskImageObj: null,
};

// DOM Element Selectors
const elements = {
  statusBadge: document.getElementById('system-status-badge'),
  statusPulse: document.getElementById('status-pulse-dot'),
  statusText: document.getElementById('status-text'),
  presetsContainer: document.getElementById('presets-container'),

  // Inputs
  plantId: document.getElementById('input-plant-id'),
  zoneId: document.getElementById('input-zone-id'),
  disease: document.getElementById('input-disease'),
  confidence: document.getElementById('input-confidence'),
  imagePath: document.getElementById('input-image-path'),
  obsDate: document.getElementById('input-obs-date'),
  urgencySlider: document.getElementById('input-urgency'),
  urgencyDisplay: document.getElementById('urgency-val-display'),
  dropzone: document.getElementById('upload-dropzone'),
  fileInput: document.getElementById('file-input'),
  dropFilename: document.getElementById('drop-filename'),
  fusionButtons: document.querySelectorAll('.fusion-btn'),
  runButton: document.getElementById('run-assessment-btn'),
  btnSpinner: document.getElementById('btn-spinner'),
  btnLabel: document.getElementById('btn-label'),

  // Microclimate strip
  valTemp: document.getElementById('val-temp'),
  valHumidity: document.getElementById('val-humidity'),
  valWetness: document.getElementById('val-wetness'),
  valWetdays: document.getElementById('val-wetdays'),

  // Canvas Viewer
  viewCanvas: document.getElementById('view-canvas'),
  canvasLoader: document.getElementById('canvas-loader'),
  opacitySlider: document.getElementById('mask-opacity-slider'),
  opacityLabel: document.getElementById('opacity-label'),

  // Morphometrics
  morphScope: document.getElementById('morph-scope'),
  morphLesions: document.getElementById('morph-lesions'),
  morphCoalescence: document.getElementById('morph-coalescence'),
  morphAspect: document.getElementById('morph-aspect'),

  // Decision & Forecasting
  scoreRing: document.getElementById('score-ring-bar'),
  scoreVal: document.getElementById('score-val-large'),
  categoryBadge: document.getElementById('category-badge'),
  stoverPill: document.getElementById('stover-grade-pill'),
  impactSub: document.getElementById('impact-sub'),
  prescriptionText: document.getElementById('prescription-text'),
  confidenceFlag: document.getElementById('confidence-flag-pill'),
  forecastScore: document.getElementById('forecast-score-val'),
  forecastCategory: document.getElementById('forecast-cat-val'),
  progressionTrend: document.getElementById('progression-trend-val'),
  weeklyRatePill: document.getElementById('weekly-rate-pill'),
  timelineCanvas: document.getElementById('timeline-chart'),

  // Zero-Collapse Lab
  labSlider: document.getElementById('lab-urgency-slider'),
  labUrgencyDisplay: document.getElementById('lab-urgency-display'),
  barFmult: document.getElementById('bar-fmult'),
  barFhybrid: document.getElementById('bar-fhybrid'),
  valFmult: document.getElementById('lab-fmult-val'),
  valFhybrid: document.getElementById('lab-fhybrid-val'),
  statusFmult: document.getElementById('fmult-status-tag'),
  statusFhybrid: document.getElementById('fhybrid-status-tag'),
};

/**
 * Initialize Application
 */
async function initApp() {
  setupEventListeners();
  await checkHealth();
  await loadPresets();
}

/**
 * Health Check Verification
 */
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('Healthcheck failed');
    const data = await res.json();
    const models = data.models_loaded || {};
    const unetOk = models.unet !== false;
    const xgbOk = models.xgboost !== false;
    const gruOk = models.gru !== false;

    if (unetOk && xgbOk && gruOk) {
      elements.statusText.textContent = 'Models Loaded (U-Net · XGB · GRU)';
      elements.statusPulse.className = 'status-pulse green';
    } else {
      elements.statusText.textContent = 'Partial Ready (Classical Fallback)';
      elements.statusPulse.className = 'status-pulse amber';
    }
  } catch (err) {
    elements.statusText.textContent = 'API Offline';
    elements.statusPulse.className = 'status-pulse crimson';
  }
}

/**
 * Load Demo Presets
 */
async function loadPresets() {
  try {
    const res = await fetch('/api/v1/demo-presets');
    if (!res.ok) throw new Error('Failed to load presets');
    const data = await res.json();
    state.presets = data.presets || [];

    renderPresetsCarousel();
    if (state.presets.length > 0) {
      selectPreset(state.presets[0]);
    }
  } catch (err) {
    elements.presetsContainer.innerHTML = '<div class="preset-error">Failed to load presets from server.</div>';
  }
}

/**
 * Render Preset Cards Carousel
 */
function renderPresetsCarousel() {
  elements.presetsContainer.innerHTML = '';
  state.presets.forEach((preset, index) => {
    const card = document.createElement('div');
    card.className = `preset-card ${index === 0 ? 'active' : ''}`;
    card.dataset.id = preset.id;
    card.innerHTML = `
      <div class="preset-card-top">
        <span class="preset-name">${preset.name}</span>
        <span class="preset-badge">${preset.badge || 'Sample'}</span>
      </div>
      <p class="preset-desc">${preset.description}</p>
      <div class="preset-meta-row">
        <span class="preset-meta-pill" title="Plant Identifier & Zone">🆔 <strong>${preset.plant_id}</strong> (${preset.zone_id})</span>
        <span class="preset-meta-pill" title="Observation Date">📅 ${preset.observation_date}</span>
        <span class="preset-meta-pill" title="Microclimate Urgency Score">⚡ <strong>${preset.urgency_score}%</strong> Urgency</span>
        <span class="preset-meta-pill" title="Weather Conditions">🌡️ ${preset.temperature_c}°C · 💧 ${preset.relative_humidity_pct}% · 🍃 ${preset.leaf_wetness_hours}h</span>
      </div>
    `;
    card.addEventListener('click', () => selectPreset(preset));
    elements.presetsContainer.appendChild(card);
  });
}

/**
 * Select and Apply a Preset
 */
function selectPreset(preset) {
  state.activePreset = preset;
  state.customImageBase64 = null; // reset custom upload on preset switch

  // Update Active Card UI
  document.querySelectorAll('.preset-card').forEach(c => {
    c.classList.toggle('active', c.dataset.id === preset.id);
  });

  // Populate Form Fields
  elements.plantId.value = preset.plant_id;
  elements.zoneId.value = preset.zone_id;
  elements.disease.value = preset.disease_label;
  elements.confidence.value = preset.disease_confidence || 0.95;
  elements.imagePath.value = preset.image_path;
  elements.dropFilename.textContent = `Preset: ${preset.image_path.split('/').pop()}`;
  if (elements.obsDate) {
    elements.obsDate.value = preset.observation_date || '2026-09-29';
  }

  elements.urgencySlider.value = preset.urgency_score;
  elements.urgencyDisplay.textContent = `${preset.urgency_score}%`;

  if (preset.temperature_c) elements.valTemp.textContent = preset.temperature_c;
  if (preset.relative_humidity_pct) elements.valHumidity.textContent = preset.relative_humidity_pct;
  if (preset.leaf_wetness_hours) elements.valWetness.textContent = preset.leaf_wetness_hours;
  if (preset.consecutive_wet_days !== undefined) elements.valWetdays.textContent = preset.consecutive_wet_days;

  // Auto trigger assessment
  runAssessment();
}

/**
 * Setup UI Event Listeners
 */
function setupEventListeners() {
  // Urgency Slider
  elements.urgencySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    elements.urgencyDisplay.textContent = `${val}%`;
    // dynamically estimate microclimate metrics
    elements.valHumidity.textContent = Math.round(50 + (val * 0.45));
    elements.valWetness.textContent = Math.round(val * 0.16);
    elements.valWetdays.textContent = Math.round(val * 0.06);
  });

  // Mask Opacity Slider
  elements.opacitySlider.addEventListener('input', (e) => {
    state.maskOpacity = parseInt(e.target.value, 10);
    elements.opacityLabel.textContent = `${state.maskOpacity}% Mask`;
    renderCanvasViewer();
  });

  // Fusion Strategy Buttons
  elements.fusionButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.fusionButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedFusionStrategy = btn.dataset.strategy;
      runAssessment();
    });
  });

  // Run Assessment Button
  elements.runButton.addEventListener('click', () => runAssessment());

  // Dropzone Click & Drag
  elements.dropzone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', handleFileSelect);

  elements.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropzone.style.borderColor = 'var(--emerald)';
  });
  elements.dropzone.addEventListener('dragleave', () => {
    elements.dropzone.style.borderColor = '';
  });
  elements.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropzone.style.borderColor = '';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImageFile(e.dataTransfer.files[0]);
    }
  });

  // Zero-Collapse Stress Test Slider
  elements.labSlider.addEventListener('input', (e) => {
    updateZeroCollapseLab(parseFloat(e.target.value));
  });

  // Responsive redraw for Timeline progression chart
  window.addEventListener('resize', () => {
    if (state.currentHistory && state.currentAssessment) {
      drawTimelineChart(state.currentHistory, state.currentAssessment.assessment || state.currentAssessment);
    }
  });
}

/**
 * File Selection Handler
 */
function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    processImageFile(e.target.files[0]);
  }
}

function processImageFile(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    state.customImageBase64 = event.target.result;
    elements.dropFilename.textContent = `Custom: ${file.name}`;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    runAssessment();
  };
  reader.readAsDataURL(file);
}

/**
 * Run Multimodal Assessment Pipeline
 */
async function runAssessment() {
  setLoadingState(true);

  const plantId = elements.plantId.value.trim() || 'REAL_PLANT_TR_001';
  const zoneId = elements.zoneId.value;
  const disease = elements.disease.value;
  const confidence = parseFloat(elements.confidence.value) || 0.95;
  const imagePath = elements.imagePath.value;
  const urgency = parseFloat(elements.urgencySlider.value) || 0.0;
  const today = new Date().toISOString().split('T')[0];
  const obsDate = (elements.obsDate && elements.obsDate.value) ? elements.obsDate.value : (state.activePreset?.observation_date || today);

  const payload = {
    vision_input: {
      plant_id: plantId,
      zone_id: zoneId,
      leaf_id: 'LEAF_01',
      observation_date: obsDate,
      disease_label: disease,
      disease_confidence: confidence,
      image_path: imagePath,
    },
    environmental_input: {
      zone_id: zoneId,
      observation_date: obsDate,
      urgency_score: urgency,
    },
    fusion_method: state.selectedFusionStrategy,
    image_base64: state.customImageBase64,
  };

  try {
    const res = await fetch('/api/v1/assess-interactive', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Assessment request failed');
    }

    const data = await res.json();
    state.currentAssessment = data;
    state.currentHistory = data.history;
    state.currentImpactScore = data.assessment.impact_score;

    // Load Images into Canvas
    await loadImagesForCanvas(data.image_base64, data.mask_base64);

    // Update UI Panels
    updateMorphometricsPanel(data.morphometrics);
    updateDecisionPanel(data.assessment);
    drawTimelineChart(data.history, data.assessment);
    updateZeroCollapseLab(parseFloat(elements.labSlider.value));

  } catch (err) {
    alert(`Assessment Error: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

/**
 * Set UI Loading Indicators
 */
function setLoadingState(isLoading) {
  elements.btnSpinner.style.display = isLoading ? 'inline-block' : 'none';
  elements.btnLabel.textContent = isLoading ? 'Segmenting & Fusing...' : '✨ Run Multimodal Assessment';
  elements.canvasLoader.style.display = isLoading ? 'flex' : 'none';
  elements.runButton.disabled = isLoading;
}

/**
 * Load Base64 Images for Canvas Viewer
 */
function loadImagesForCanvas(rawB64, maskB64) {
  return new Promise((resolve) => {
    let loadedCount = 0;
    const checkDone = () => {
      loadedCount++;
      if (loadedCount === 2) {
        renderCanvasViewer();
        resolve();
      }
    };

    const rawImg = new Image();
    rawImg.onload = checkDone;
    rawImg.src = rawB64;
    state.rawImageObj = rawImg;

    const maskImg = new Image();
    maskImg.onload = checkDone;
    maskImg.src = maskB64;
    state.maskImageObj = maskImg;
  });
}

/**
 * Render Canvas Viewer with Dual Layer Blending
 */
function renderCanvasViewer() {
  const canvas = elements.viewCanvas;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Raw Leaf Photo
  if (state.rawImageObj) {
    ctx.globalAlpha = 1.0;
    ctx.drawImage(state.rawImageObj, 0, 0, canvas.width, canvas.height);
  }

  // 2. Draw Colorized Mask with Alpha Slider
  if (state.maskImageObj && state.maskOpacity > 0) {
    ctx.globalAlpha = state.maskOpacity / 100.0;
    ctx.drawImage(state.maskImageObj, 0, 0, canvas.width, canvas.height);
  }

  ctx.globalAlpha = 1.0;
}

/**
 * Update Morphometrics Strip (FR3)
 */
function updateMorphometricsPanel(morph) {
  if (!morph) return;
  elements.morphScope.textContent = `${morph.scope_pct}%`;
  elements.morphLesions.textContent = morph.num_lesions;
  elements.morphCoalescence.textContent = `${morph.coalescence_ratio_pct}%`;
  elements.morphAspect.textContent = morph.leaf_aspect_ratio;
}

/**
 * Update Decision & Severity Summary Panel (FR6, FR7, FR10, FR11)
 */
function updateDecisionPanel(assessment) {
  const score = Math.round(assessment.severity_score * 10) / 10;
  elements.scoreVal.textContent = score;

  // Animate SVG Ring
  // Circumference = 2 * PI * 50 ≈ 314.15
  const circumference = 314.15;
  const offset = circumference - (score / 100.0) * circumference;
  elements.scoreRing.style.strokeDashoffset = offset;

  // Severity Category Badge & Ring Color
  const cat = assessment.severity_category;
  elements.categoryBadge.textContent = cat;

  let ringColor = 'var(--emerald)';
  let badgeClass = 'badge-low';

  if (cat === 'Critical') {
    ringColor = 'var(--crimson)';
    badgeClass = 'badge-critical';
  } else if (cat === 'High') {
    ringColor = '#f97316';
    badgeClass = 'badge-high';
  } else if (cat === 'Moderate') {
    ringColor = 'var(--amber)';
    badgeClass = 'badge-moderate';
  }

  elements.scoreRing.style.stroke = ringColor;
  elements.categoryBadge.className = `category-badge ${badgeClass}`;

  // Stover & Impact Subtitle
  const grade = assessment.stover_grade !== null ? assessment.stover_grade : '--';
  elements.stoverPill.textContent = `Stover-Gauhl: Grade ${grade}`;
  elements.impactSub.textContent = `Bio Impact: ${Math.round(assessment.impact_score)} · Urgency: ${Math.round(assessment.urgency_score)}% · Fusion: ${assessment.fusion_method_used}`;

  // Prescription Protocol
  elements.prescriptionText.textContent = assessment.recommended_action;

  // Forecast Card
  elements.confidenceFlag.textContent = assessment.confidence_flag;
  if (assessment.forecast_7d_score !== null && assessment.forecast_7d_score !== undefined) {
    elements.forecastScore.textContent = Math.round(assessment.forecast_7d_score * 10) / 10;
    elements.forecastCategory.textContent = assessment.forecast_7d_category || '--';
  } else {
    elements.forecastScore.textContent = 'Cold Start';
    elements.forecastCategory.textContent = 'Need ≥ 2 visits';
  }

  elements.progressionTrend.textContent = assessment.progression_trend;
  if (assessment.weekly_rate_of_change !== null) {
    const rate = Math.round(assessment.weekly_rate_of_change * 10) / 10;
    elements.weeklyRatePill.textContent = `Rate: ${rate > 0 ? '+' : ''}${rate} pts/7d`;
  } else {
    elements.weeklyRatePill.textContent = 'Rate: Baseline';
  }
}

/**
 * Draw Longitudinal Timeline Chart on HTML5 Canvas
 */
function drawTimelineChart(history, assessment) {
  const canvas = elements.timelineCanvas;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  // Scale for retina displays
  const w = canvas.parentElement.clientWidth;
  const h = 140;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, w, h);

  if (!history || history.length === 0) {
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No previous visits recorded for this plant.', w / 2, h / 2);
    return;
  }

  const padding = { top: 22, right: 38, bottom: 25, left: 35 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;

  // Deduplicate history by observation_date (latest record per date)
  const dateMap = new Map();
  history.forEach(r => {
    const d = r.observation_date || '';
    const score = parseFloat(r.severity_score);
    if (!isNaN(score)) {
      dateMap.set(d, {
        date: d,
        score: score,
        label: d.length >= 10 ? d.substring(5) : d,
      });
    }
  });

  // Sort chronologically
  let points = Array.from(dateMap.values()).sort((a, b) => a.date.localeCompare(b.date));

  // Limit display to the most recent 7 visits to ensure spacious, readable layout
  if (points.length > 7) {
    points = points.slice(-7);
  }

  // Add forecast point if available
  const hasForecast = assessment.forecast_7d_score !== null && assessment.forecast_7d_score !== undefined;
  if (hasForecast) {
    points.push({
      date: 'Forecast (+7d)',
      score: parseFloat(assessment.forecast_7d_score),
      label: '+7d (GRU)',
      isForecast: true,
    });
  }

  const n = points.length;
  if (n === 1) {
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Single visit logged (Score: ${points[0].score.toFixed(1)}). Next visit enables GRU trend.`, w / 2, h / 2);
    return;
  }

  const getX = (i) => padding.left + (i / (n - 1)) * plotW;
  const getY = (val) => padding.top + plotH - (Math.min(100, Math.max(0, val)) / 100) * plotH;

  // Grid Lines
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
  ctx.lineWidth = 1;
  [0, 25, 50, 75, 100].forEach(tick => {
    const y = getY(tick);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(tick, padding.left - 6, y + 3);
  });

  // Historical Area Fill & Solid Line
  const histPoints = hasForecast ? points.slice(0, -1) : points;
  if (histPoints.length > 1) {
    // Gradient Area
    const areaGrad = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
    areaGrad.addColorStop(0, 'rgba(5, 150, 105, 0.15)');
    areaGrad.addColorStop(1, 'rgba(5, 150, 105, 0.0)');

    ctx.beginPath();
    histPoints.forEach((p, idx) => {
      const x = getX(idx);
      const y = getY(p.score);
      if (idx === 0) {
        ctx.moveTo(x, padding.top + plotH);
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.lineTo(getX(histPoints.length - 1), padding.top + plotH);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Solid Line
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#059669';
    histPoints.forEach((p, idx) => {
      const x = getX(idx);
      const y = getY(p.score);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Forecast Dashed Line
  if (hasForecast && histPoints.length > 0) {
    const lastHist = histPoints[histPoints.length - 1];
    const fcPoint = points[points.length - 1];
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#2563eb';
    ctx.moveTo(getX(histPoints.length - 1), getY(lastHist.score));
    ctx.lineTo(getX(points.length - 1), getY(fcPoint.score));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw Dots & Value Labels
  points.forEach((p, idx) => {
    const x = getX(idx);
    const y = getY(p.score);

    // Outer glow for forecast / active
    if (p.isForecast || idx === histPoints.length - 1) {
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = p.isForecast ? 'rgba(37, 99, 235, 0.15)' : 'rgba(5, 150, 105, 0.15)';
      ctx.fill();
    }

    // Dot
    ctx.beginPath();
    ctx.arc(x, y, p.isForecast ? 5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = p.isForecast ? '#2563eb' : '#059669';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Text Label with light pill backdrop for maximum legibility
    const valText = p.score.toFixed(1);
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';

    const textY = y < 28 ? y + 16 : y - 8;
    const textW = ctx.measureText(valText).width;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x - textW / 2 - 4, textY - 9, textW + 8, 12, 3);
    } else {
      ctx.fillRect(x - textW / 2 - 4, textY - 9, textW + 8, 12);
    }
    ctx.fill();
    ctx.strokeStyle = 'rgba(203, 213, 225, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = p.isForecast ? '#1d4ed8' : '#0f172a';
    ctx.fillText(valText, x, textY);

    // X-axis label
    ctx.fillStyle = p.isForecast ? '#2563eb' : '#64748b';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(p.label, x, h - 8);
  });
}

/**
 * Update Zero-Collapse Stress Test Simulator
 */
function updateZeroCollapseLab(urgency) {
  elements.labUrgencyDisplay.textContent = `${Math.round(urgency)}%`;

  const impact = state.currentImpactScore || 50.0;
  const uFrac = urgency / 100.0;

  // Unconstrained Multiplicative (Literature Flaw)
  const fMult = impact * uFrac;

  // Damped Multiplicative Hybrid (Our Safe Formula)
  const fHybrid = impact * (0.70 + 0.30 * uFrac);

  elements.barFmult.style.width = `${Math.min(100, Math.max(0, fMult))}%`;
  elements.barFhybrid.style.width = `${Math.min(100, Math.max(0, fHybrid))}%`;

  elements.valFmult.textContent = `${fMult.toFixed(1)} / 100`;
  elements.valFhybrid.textContent = `${fHybrid.toFixed(1)} / 100`;

  if (urgency <= 15) {
    elements.statusFmult.textContent = '⚠️ CATASTROPHIC COLLAPSE';
    elements.statusFmult.className = 'failure-indicator';
  } else {
    elements.statusFmult.textContent = 'Unconstrained Multiplicative';
    elements.statusFmult.className = 'failure-indicator';
  }

  elements.statusFhybrid.textContent = `✅ ${Math.round(0.70 * impact)}% BIOLOGICAL FLOOR SAFE`;
}

// Start application on DOM Ready
document.addEventListener('DOMContentLoaded', initApp);
