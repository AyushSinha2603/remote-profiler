SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# Material Properties (Reference: IRC 15:2017 - Road Maintenance, IS 2386 for aggregates)
HMA_DENSITY = 2.4          # kg per liter (hot-mix asphalt ~2300-2500 kg/m3)
TACK_COAT_RATE = 0.3       # liters per m2 (bitumen emulsion application rate: 0.2-0.4 L/m2)
AGGREGATE_DENSITY = 1600   # kg/m3 (granular base material per IS standards)
BASE_REPAIR_DEPTH = 0.05   # meters — asphalt layer above aggregate

# Severity-based Material Multipliers (adjusted for repair method intensity)
# These multipliers adjust base material quantities based on damage severity
# Reference: IS 456, IRC guidelines, and empirical road repair data
MATERIAL_MULTIPLIERS = {
    SEVERITY_LOW: {
        "hotmix_multiplier": 0.6,      # Reduced material for surface-only patch
        "tack_coat_multiplier": 0.8,   # Minimal tack coat
        "aggregate_multiplier": 0.0,   # No base layer repair
        "labour_multiplier": 0.5,
        "equipment_multiplier": 0.4,
    },
    SEVERITY_MEDIUM: {
        "hotmix_multiplier": 1.0,      # Standard patch
        "tack_coat_multiplier": 1.0,
        "aggregate_multiplier": 0.3,   # Partial base repair
        "labour_multiplier": 1.0,
        "equipment_multiplier": 0.8,
    },
    SEVERITY_HIGH: {
        "hotmix_multiplier": 1.3,      # Full-depth patch
        "tack_coat_multiplier": 1.2,
        "aggregate_multiplier": 0.7,   # Significant base repair
        "labour_multiplier": 1.3,
        "equipment_multiplier": 1.0,
    },
    SEVERITY_CRITICAL: {
        "hotmix_multiplier": 1.6,      # Full-depth with base rebuild
        "tack_coat_multiplier": 1.5,
        "aggregate_multiplier": 1.0,   # Full base layer replacement
        "labour_multiplier": 1.6,
        "equipment_multiplier": 1.2,
    },
}

# Per-unit costs (INR) - Updated based on market rates and IS cost indices
COST_HMA_PER_KG = 7.0              # Hot-mix asphalt cost
COST_TACK_PER_LITER = 60.0         # Bitumen emulsion cost
COST_AGGREGATE_PER_KG = 1.0        # Base aggregates cost
COST_LABOUR = 300                  # Base labour cost per repair
COST_EQUIPMENT = 150               # Base equipment cost per repair

REPAIR_METHODS = {
    SEVERITY_LOW: "Surface patch / slurry seal",
    SEVERITY_MEDIUM: "Throw-and-roll patch",
    SEVERITY_HIGH: "Full-depth semi-permanent patch",
    SEVERITY_CRITICAL: "Full-depth patch with base repair",
}


def classify_severity(area_m2: float, depth_m: float) -> str:
    if depth_m < 0.025 and area_m2 < 0.05:
        return SEVERITY_LOW
    if depth_m < 0.05 and area_m2 < 0.15:
        return SEVERITY_MEDIUM
    if depth_m < 0.1 and area_m2 < 0.3:
        return SEVERITY_HIGH
    return SEVERITY_CRITICAL


def estimate_materials(area_m2: float, depth_m: float, volume_liters: float, severity: str = None) -> dict:
    """
    Estimate materials needed for pothole repair.
    
    Args:
        area_m2: Surface area in square meters
        depth_m: Pothole depth in meters
        volume_liters: Calculated volume in liters
        severity: Optional severity level (LOW/MEDIUM/HIGH/CRITICAL).
                  If None, will be auto-calculated.
    
    Returns:
        Dictionary with material quantities
    """
    # Auto-classify severity if not provided
    if severity is None:
        severity = classify_severity(area_m2, depth_m)
    
    # Get multipliers for this severity level
    multipliers = MATERIAL_MULTIPLIERS.get(severity, MATERIAL_MULTIPLIERS[SEVERITY_MEDIUM])
    
    # Base material calculations
    hotmix_kg = round(volume_liters * HMA_DENSITY, 2)
    tack_coat_liters = round(area_m2 * TACK_COAT_RATE, 3)
    aggregate_base_kg = 0.0

    if depth_m > 0.1:
        aggregate_depth = depth_m - BASE_REPAIR_DEPTH
        aggregate_base_kg = round(aggregate_depth * area_m2 * AGGREGATE_DENSITY, 2)

    # Apply severity-based multipliers
    hotmix_kg = round(hotmix_kg * multipliers["hotmix_multiplier"], 2)
    tack_coat_liters = round(tack_coat_liters * multipliers["tack_coat_multiplier"], 3)
    aggregate_base_kg = round(aggregate_base_kg * multipliers["aggregate_multiplier"], 2)

    return {
        "hotmix_kg": hotmix_kg,
        "tack_coat_liters": tack_coat_liters,
        "aggregate_base_kg": aggregate_base_kg,
        "severity": severity,
    }


def estimate_cost(materials: dict, severity: str = None) -> float:
    """
    Estimate total repair cost based on materials and severity.
    
    Args:
        materials: Dictionary with material quantities (from estimate_materials)
        severity: Optional severity level for labour/equipment adjustments
    
    Returns:
        Total estimated cost in INR
    """
    if severity is None:
        severity = materials.get("severity", SEVERITY_MEDIUM)
    
    multipliers = MATERIAL_MULTIPLIERS.get(severity, MATERIAL_MULTIPLIERS[SEVERITY_MEDIUM])
    
    cost = (COST_LABOUR * multipliers["labour_multiplier"] + 
            COST_EQUIPMENT * multipliers["equipment_multiplier"])
    cost += materials["hotmix_kg"] * COST_HMA_PER_KG
    cost += materials["tack_coat_liters"] * COST_TACK_PER_LITER
    cost += materials["aggregate_base_kg"] * COST_AGGREGATE_PER_KG
    return round(cost, 2)


def estimate_repair(area_m2: float, depth_m: float, volume_m3: float, volume_liters: float) -> dict:
    severity = classify_severity(area_m2, depth_m)
    materials = estimate_materials(area_m2, depth_m, volume_liters, severity)
    cost = estimate_cost(materials, severity)

    return {
        "severity": severity,
        "repair_method": REPAIR_METHODS[severity],
        "materials": materials,
        "estimated_cost_inr": cost,
    }


# ============ ML Integration (Active) ============
# Trained ML models stored in backend/models/material_*.pkl
# Train with: python utils/ml_trainer.py --train data/repair_data.csv
# Models predict material quantities based on pothole dimensions

_ml_models = {}  # Cache loaded models
_ml_enabled = False

def enable_ml_mode(model_dir: str = 'backend/models'):
    """
    Enable ML-based material prediction.
    
    Call this once during app initialization:
        from utils.material_estimator import enable_ml_mode
        enable_ml_mode()
    
    Args:
        model_dir: Directory containing trained model files
    
    Returns:
        bool: True if models loaded successfully, False otherwise
    """
    global _ml_models, _ml_enabled
    
    try:
        import joblib
        from pathlib import Path
        
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            print(f"⚠ ML models directory not found: {model_dir}")
            return False
        
        # Load individual models for each material type
        model_files = {
            'hotmix_kg': f"{model_dir}/material_hotmix_kg.pkl",
            'tack_coat_liters': f"{model_dir}/material_tack_coat_liters.pkl",
            'aggregate_base_kg': f"{model_dir}/material_aggregate_base_kg.pkl",
        }
        
        scaler_files = {
            'hotmix_kg': f"{model_dir}/scaler_hotmix_kg.pkl",
            'tack_coat_liters': f"{model_dir}/scaler_tack_coat_liters.pkl",
            'aggregate_base_kg': f"{model_dir}/scaler_aggregate_base_kg.pkl",
        }
        
        for material, model_file in model_files.items():
            try:
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_files[material])
                _ml_models[material] = {'model': model, 'scaler': scaler}
            except FileNotFoundError:
                print(f"⚠ Model file not found: {model_file}")
                return False
        
        _ml_enabled = True
        print("✓ ML models loaded successfully")
        return True
        
    except ImportError:
        print("⚠ joblib not installed. Cannot enable ML mode.")
        return False
    except Exception as e:
        print(f"⚠ Failed to load ML models: {e}")
        return False


def predict_materials_ml(area_m2: float, depth_m: float, volume_liters: float) -> dict:
    """
    Use trained ML model to predict material quantities.
    Automatically falls back to rule-based estimation if ML unavailable.
    
    Args:
        area_m2: Surface area in square meters
        depth_m: Pothole depth in meters
        volume_liters: Calculated volume in liters
    
    Returns:
        Dictionary with predicted material quantities and source
    """
    if not _ml_enabled or not _ml_models:
        # Fallback to rule-based estimation
        severity = classify_severity(area_m2, depth_m)
        materials = estimate_materials(area_m2, depth_m, volume_liters, severity)
        materials['prediction_source'] = 'RULE_BASED'
        return materials
    
    try:
        import numpy as np
        
        # Prepare input features
        features = np.array([[area_m2, depth_m, volume_liters]])
        
        predictions = {}
        for material, models_dict in _ml_models.items():
            scaler = models_dict['scaler']
            model = models_dict['model']
            
            # Scale features and predict
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            
            # Ensure non-negative predictions
            predictions[material] = max(0, float(pred))
        
        result = {
            "hotmix_kg": round(predictions['hotmix_kg'], 2),
            "tack_coat_liters": round(predictions['tack_coat_liters'], 3),
            "aggregate_base_kg": round(predictions['aggregate_base_kg'], 2),
            "severity": classify_severity(area_m2, depth_m),
            "prediction_source": "ML_MODEL",
        }
        
        return result
        
    except Exception as e:
        print(f"⚠ ML prediction failed: {e}. Falling back to rule-based estimation.")
        severity = classify_severity(area_m2, depth_m)
        materials = estimate_materials(area_m2, depth_m, volume_liters, severity)
        materials['prediction_source'] = 'RULE_BASED (ML failed)'
        return materials
