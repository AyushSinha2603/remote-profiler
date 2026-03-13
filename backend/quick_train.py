#!/usr/bin/env python3
"""
Quick-start script to train and test ML model for material estimation.

Usage:
    python backend/quick_train.py
"""

import os
import sys
from pathlib import Path

# Ensure we can import utils
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("🚀 QUICK-START: ML MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Check if sample data exists
    sample_data = Path(__file__).parent / 'data' / 'sample_repair_data.csv'
    if not sample_data.exists():
        print(f"❌ Sample data not found: {sample_data}")
        print("   Please ensure sample_repair_data.csv exists in backend/data/")
        return False
    
    print(f"\n✓ Found sample data: {sample_data}")
    print(f"  Size: {sample_data.stat().st_size} bytes")
    
    # Step 2: Train ML model
    print("\n📊 Training ML model...")
    try:
        from utils.ml_trainer import MaterialEstimatorML
        
        trainer = MaterialEstimatorML(model_type='random_forest')
        df = trainer.load_data(str(sample_data))
        trainer.train(df, test_size=0.2)
        
        # Save models
        model_dir = Path(__file__).parent / 'models'
        trainer.save_model(str(model_dir))
        
        print(f"\n✓ Models saved to: {model_dir}")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install scikit-learn pandas numpy joblib")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Step 3: Test predictions
    print("\n🧪 Testing predictions...")
    try:
        from utils.material_estimator import enable_ml_mode, predict_materials_ml
        
        if enable_ml_mode(str(model_dir)):
            print("\n--- Test Cases ---")
            test_cases = [
                (0.05, 0.02, 0.1, "Small LOW pothole"),
                (0.15, 0.05, 0.5, "Medium MEDIUM pothole"),
                (0.3, 0.1, 2.0, "Large HIGH pothole"),
                (0.5, 0.15, 3.0, "Critical CRITICAL pothole"),
            ]
            
            for area, depth, volume, desc in test_cases:
                pred = predict_materials_ml(area, depth, volume)
                print(f"\n{desc}")
                print(f"  Input: area={area}m², depth={depth}m, volume={volume}L")
                print(f"  Predictions:")
                print(f"    HMA: {pred['hotmix_kg']}kg")
                print(f"    Tack: {pred['tack_coat_liters']}L")
                print(f"    Aggregate: {pred['aggregate_base_kg']}kg")
                print(f"    Severity: {pred['severity']}")
                print(f"    Source: {pred['prediction_source']}")
        else:
            print("❌ Failed to load ML models")
            return False
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! ML model is trained and working")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Collect real repair data (20+ records)")
    print("   2. Save to backend/data/repair_data.csv")
    print("   3. Retrain: python backend/utils/ml_trainer.py --train backend/data/repair_data.csv")
    print("   4. Start Flask app: python backend/app.py")
    print("\n📖 For full guide, see: backend/ML_TRAINING_GUIDE.md")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
