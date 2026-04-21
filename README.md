# VisionCred: AI Credit Engine for Kirana Stores

> **Vision + Geo Intelligence + Economic Modeling = Explainable Credit Assessment**

An AI-powered credit assessment engine that estimates a kirana (grocery) store's cash flow using store images and GPS location data. Built with explainability first — no black-box ML, only transparent formulas and rule-based logic.

---

## 🏗️ Project Structure

```
tensorx/
├── main.py                    # CLI entry point (batch + single store)
├── requirements.txt           # Python dependencies
├── README.md
│
├── src/                       # Core modules
│   ├── __init__.py
│   ├── config.py              # All constants, thresholds, parameters
│   ├── vision.py              # YOLOv8 object detection & feature extraction
│   ├── geo_intel.py           # GPS-based location intelligence
│   ├── features.py            # Feature engineering & consolidation
│   ├── economic_model.py      # Transparent cash flow formulas
│   ├── fraud_detection.py     # Anomaly & fraud detection
│   ├── confidence.py          # Confidence scoring
│   └── pipeline.py            # End-to-end pipeline orchestrator
│
├── app/                       # Frontend
│   ├── __init__.py
│   └── gradio_app.py          # Gradio web interface
│
├── models/                    # Model weights (auto-downloaded)
│   └── __init__.py
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── logger.py              # Centralized logging
│   └── data_loader.py         # Store data loading & validation
│
├── data/                      # Dataset
│   └── raw/
│       ├── store_1/
│       │   ├── shelf_1.webp
│       │   ├── shelf_2.webp
│       │   ├── shelf_3.webp
│       │   ├── counter.webp
│       │   ├── outside.webp
│       │   └── metadata.json
│       ├── store_2/
│       └── ...
│
└── outputs/                   # Generated reports
    ├── sample_output.json
    ├── batch_results.json
    └── summary.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (All Stores)

```bash
python main.py
```

### 3. Run Single Store

```bash
python main.py --store store_1
```

### 4. Run Multiple Specific Stores

```bash
python main.py --store store_1 store_3 store_7
```

### 5. Launch Gradio Web App

```bash
python app/gradio_app.py
```

Then open **http://localhost:7860** in your browser.

---

## 📊 Pipeline Flow

```
Store Images (.webp)  →  YOLOv8 Vision Analysis
                              ↓
                         Feature Extraction
                         (shelf density, SKU diversity, inventory value)
                              ↓
metadata.json (GPS)   →  Geo Intelligence
                         (footfall, competition, location type)
                              ↓
                         Feature Engineering
                         (consolidation + derived features)
                              ↓
                         Economic Model
                         (transparent formulas, range outputs)
                              ↓
                         Fraud Detection
                         (anomaly checks, duplicate images)
                              ↓
                         Confidence Scoring
                              ↓
                         Structured JSON Output
```

---

## 🔍 Module Details

### Vision Module (`src/vision.py`)
- Uses **YOLOv8n** (ultralytics) for object detection on store images
- Extracts: **Shelf Density Index**, **SKU Diversity Score**, **Inventory Value**
- Supports WEBP, JPG, PNG formats
- Lazy-loads model on first use

### Geo Intelligence (`src/geo_intel.py`)
- Reads `metadata.json` for GPS coordinates
- Classifies location using **Haversine distance** to metro centers
- Generates: **footfall_score**, **competition_density**, **location_type**
- Falls back to Hyderabad center coordinates if metadata is missing

### Economic Model (`src/economic_model.py`)
- **Fully transparent formula**:
  ```
  Daily Sales = (Visible Inventory × Uplift Factor)
                × Turnover Rate × Location Multiplier × Demand Factor
  ```
- All outputs are **ranges** (min, max)
- Identifies **key drivers** in plain English

### Fraud Detection (`src/fraud_detection.py`)
- Inventory-footfall mismatch detection
- Duplicate image detection (perceptual hashing)
- SKU anomaly detection
- Metadata quality checks
- Image quality assessment
- Mandatory view coverage checks (shelf/counter/outside)

### Confidence Scoring (`src/confidence.py`)
- Weighted combination of 4 components:
  - Image count (25%)
  - Detection quality (35%)
  - Consistency (20%)
  - Metadata presence (20%)

---

## 📋 Output Format

```json
{
  "store_id": "store_1",
  "daily_sales_range": [6000, 9000],
  "monthly_revenue_range": [180000, 270000],
  "monthly_income_range": [25000, 45000],
  "confidence_score": 0.72,
  "risk_flags": ["inventory_footfall_mismatch", "limited_view_coverage"],
  "key_drivers": [...],
  "recommendation": "needs_verification"
}
```

See `outputs/sample_output.json` for a comprehensive example.

---

## ⚙️ Configuration

All parameters are documented in `src/config.py`:
- YOLO thresholds
- Inventory pricing assumptions
- Geo distance thresholds
- Economic model parameters (turnover rate, margins)
- Fraud detection thresholds
- Confidence weights

---

## 🎯 Design Principles

1. **Explainability**: Every number traces back to a documented formula
2. **Range-based**: No single-value predictions — always min/max ranges
3. **Modular**: Each module is independently testable
4. **Robust**: Graceful handling of missing data, bad images, no GPS
5. **Hackathon-ready**: Clean output, visual demos, comprehensive logging
