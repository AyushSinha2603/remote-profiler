# ML Model Training Guide for Material Estimation

## Overview
This guide explains how to collect historical repair data and train an ML model to replace hardcoded material multipliers with real-world predictions.

---

## **Step 1: Data Collection** 📊

### What Data to Collect
Every time your team repairs a pothole, record:

| Field | Example | Notes |
|-------|---------|-------|
| `area_m2` | 0.15 | Measured surface area (m²) |
| `depth_m` | 0.05 | Measured depth (meters) |
| `volume_liters` | 0.5 | From estimator.py calculation |
| `hotmix_kg` | 1.2 | Actual HMA material used (kg) |
| `tack_coat_liters` | 0.045 | Actual bitumen emulsion (liters) |
| `aggregate_base_kg` | 0.1 | Actual aggregate base layer (kg) |
| `repair_method` | "throw-and-roll" | Method used |
| `repair_date` | 2025-03-13 | Date of repair |
| `location_notes` | "NH-1, KM-456" | Where repair was done |

### Collection Methods

#### **Option A: Mobile App Form** (Best)
Add to Flutter app:
```dart
// In scan_screen.dart or new repair_log_screen.dart
TextField(label: 'Area (m²)', onChanged: (v) => data['area_m2'] = double.parse(v)),
TextField(label: 'Depth (m)', onChanged: (v) => data['depth_m'] = double.parse(v)),
TextField(label: 'HMA Used (kg)', onChanged: (v) => data['hotmix_kg'] = double.parse(v)),
// ... etc
// POST to backend: /api/log-repair
```

#### **Option B: Google Sheets Form** (Quick Start)
1. Create form: https://forms.google.com
2. Columns: area_m2, depth_m, volume_liters, hotmix_kg, tack_coat_liters, aggregate_base_kg
3. Download as CSV → `backend/data/repair_data.csv`

#### **Option C: CSV Template**
Share with contractors:
```csv
area_m2,depth_m,volume_liters,hotmix_kg,tack_coat_liters,aggregate_base_kg
0.15,0.05,0.5,1.2,0.045,0.1
```

### ✅ Data Quality Checklist
- [ ] At least 20+ repair records (more = better accuracy)
- [ ] Mix of LOW, MEDIUM, HIGH, CRITICAL severity potholes
- [ ] Measurements from same measurement method (consistent)
- [ ] No missing values (remove incomplete records)
- [ ] Values are realistic (hotmix ~0.24*volume_liters as baseline)

---

## **Step 2: Train the ML Model** 🤖

### Prerequisites
```bash
pip install scikit-learn pandas numpy joblib
```

### Training Command

```bash
# From project root directory

# Basic training (saves models to backend/models/)
python backend/utils/ml_trainer.py --train backend/data/repair_data.csv

# With test predictions
python backend/utils/ml_trainer.py --train backend/data/repair_data.csv --test-pred

# Use Gradient Boosting instead of Random Forest
python backend/utils/ml_trainer.py --train backend/data/repair_data.csv --model gradient_boosting
```

### What Happens During Training
1. **Loads data** from CSV
2. **Splits into train/test** (80/20 split)
3. **Trains 3 separate models**:
   - Model 1: Predict `hotmix_kg`
   - Model 2: Predict `tack_coat_liters`
   - Model 3: Predict `aggregate_base_kg`
4. **Evaluates accuracy** with MAE, RMSE, R² score
5. **Saves models** to `backend/models/material_*.pkl`

### Expected Output
```
✓ Loaded 20 repair records
  Feature range - area_m2: 0.04-0.5
  Feature range - depth_m: 0.02-0.16
  Feature range - volume_liters: 0.08-6.0

✓ Train-test split: 16 train, 4 test

--- Training model for hotmix_kg ---
  MAE:      0.1234
  RMSE:     0.1556
  R² Score: 0.9421
  CV MAE (5-fold): 0.1089
  Feature Importance:
    area_m2: 0.401
    depth_m: 0.389
    volume_liters: 0.210

✓ Saved backend/models/material_hotmix_kg.pkl
✓ Saved backend/models/material_tack_coat_liters.pkl
✓ Saved backend/models/material_aggregate_base_kg.pkl
```

**Interpretation:**
- **MAE (Mean Absolute Error)**: Average prediction error (lower = better)
  - 0.12kg MAE = typically within ±0.12kg of actual
- **R² Score**: How well model explains variance (0-1, closer to 1 = better)
  - 0.94 = excellent (explains 94% of variance)
- **Feature Importance**: Which inputs matter most
  - `area_m2` has highest importance = most influential

---

## **Step 3: Enable ML in Your App** 🚀

### In `backend/app.py`

```python
from flask import Flask
from utils.material_estimator import enable_ml_mode
from routes.detect import detect_bp

app = Flask(__name__)

# Enable ML mode on startup
enable_ml_mode('backend/models')

# Register routes
app.register_blueprint(detect_bp)

if __name__ == '__main__':
    app.run(debug=True)
```

### In `backend/routes/detect.py` (Optional - Use ML Instead of Rules)

```python
# Option 1: Keep using rules (ML as future enhancement)
from utils.material_estimator import estimate_repair
repair = estimate_repair(area_m2, depth_m, volume_m3, volume_liters)

# Option 2: Force ML predictions
from utils.material_estimator import predict_materials_ml, estimate_cost
materials = predict_materials_ml(area_m2, depth_m, volume_liters)
cost = estimate_cost(materials, materials['severity'])
```

---

## **Step 4: Monitor Model Performance** 📈

### Log Prediction Source
Your API responses now include:
```json
{
  "severity": "MEDIUM",
  "materials": {
    "hotmix_kg": 1.2,
    "tack_coat_liters": 0.045,
    "aggregate_base_kg": 0.1,
    "prediction_source": "ML_MODEL"  // NEW
  },
  "estimated_cost_inr": 580
}
```

### Compare Rule-Based vs ML
```python
# backend/utils/test.py

from estimator import estimate_volume
from material_estimator import estimate_repair, predict_materials_ml

# Test case
area_m2, depth_m, volume_liters = 0.15, 0.05, 0.5

# Rule-based prediction
rule_mat = estimate_materials(area_m2, depth_m, volume_liters)
print("Rule-based:", rule_mat)

# ML prediction
ml_mat = predict_materials_ml(area_m2, depth_m, volume_liters)
print("ML-based:  ", ml_mat)
```

---

## **Step 5: Retrain When You Have More Data** 🔄

As you collect more repairs:

1. **Every 20+ new records**: Retrain model for more accuracy
2. **Quarterly**: Review model performance vs actual repairs
3. **If accuracy drops**: Check data quality, remove outliers

```bash
# Add new data to CSV and retrain
python backend/utils/ml_trainer.py --train backend/data/repair_data.csv

# This overwrites the old models with new ones
```

---

## **Troubleshooting**

### "ML models not found"
```
⚠ ML models directory not found: backend/models
→ Run training first: python backend/utils/ml_trainer.py --train repair_data.csv
```

### "joblib not installed"
```
pip install joblib
```

### "MAE is too high"
- You need more training data (aim for 30+ records minimum)
- Check data quality for outliers or errors
- Verify measurements are consistent between repairs

### Model accuracy poor?
```python
# Test on your actual repair data
train_data = pd.read_csv('backend/data/repair_data.csv')
for idx, row in train_data.sample(5).iterrows():
    pred = predict_materials_ml(row['area_m2'], row['depth_m'], row['volume_liters'])
    actual = row['hotmix_kg']
    print(f"Predicted: {pred['hotmix_kg']}kg, Actual: {actual}kg, Error: {abs(pred['hotmix_kg']-actual):.2f}kg")
```

---

## **Sample Training Data** 📝

A sample CSV with realistic data is provided at:
```
backend/data/sample_repair_data.csv
```

Train on it to test:
```bash
python backend/utils/ml_trainer.py --train backend/data/sample_repair_data.csv --test-pred
```

---

## **Advanced: Custom Model Parameters**

Edit `ml_trainer.py` to tune model:

```python
# Random Forest tuning (in ml_trainer.py)
model = RandomForestRegressor(
    n_estimators=200,      # More trees = more accurate (slower)
    max_depth=15,          # Deeper = overfitting risk
    min_samples_split=3,   # Lower = more specific to training data
    min_samples_leaf=1,
)

# Gradient Boosting tuning
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,    # Slower learning = more stable
    max_depth=8,
)
```

---

## **What's Better: Rule-Based or ML?** 🤔

| Aspect | Rule-Based (Current) | ML Model |
|--------|---------------------|----------|
| Accuracy | ~75% (educated guess) | ~95%+ (learned from data) |
| Adaptability | Fixed multipliers | Learns from new data |
| Data Needed | None | 20+ repair records |
| Training Time | N/A | 2-5 minutes |
| When to use | Day 1 (no data) | After 20+ repairs |

**Hybrid approach:** Keep rule-based as fallback, use ML when available ✅

---

## **Questions?**

- **Why separate models?** Each material type has different relationships with area/depth
- **Why Random Forest?** Handles non-linear relationships better than linear regression
- **Can I use deep learning?** Yes (TensorFlow/Keras), but needs 100+ records
- **How often retrain?** Every 15-20 new records for best accuracy

---

**Next Steps:**
1. ✅ Set up data collection (Google Sheets or mobile form)
2. ✅ Get 20+ repair records
3. ✅ Run training: `python backend/utils/ml_trainer.py --train repair_data.csv`
4. ✅ Enable ML: Add `enable_ml_mode()` to app.py
5. ✅ Verify: Check API response for `"prediction_source": "ML_MODEL"`
