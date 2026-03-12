"""
ML Model Training for Material Estimation

This script trains a machine learning model using historical pothole repair data
to predict material quantities more accurately than rule-based estimation.

Workflow:
1. Collect repair data (area_m2, depth_m, volume_liters, actual material usage)
2. Prepare CSV file with columns: area_m2, depth_m, volume_liters, hotmix_kg, tack_coat_liters, aggregate_base_kg
3. Run: python ml_trainer.py --train repair_data.csv
4. Save model: repair_model.pkl
5. Load in detect.py and use predict_materials_ml()
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import argparse
from pathlib import Path


class MaterialEstimatorML:
    """Train and evaluate ML models for material prediction"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize ML model trainer
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.models = {}  # Separate model for each material type
        self.scalers = {}
        self.feature_columns = ['area_m2', 'depth_m', 'volume_liters']
        self.target_columns = ['hotmix_kg', 'tack_coat_liters', 'aggregate_base_kg']
        
    def load_data(self, csv_path):
        """
        Load and validate training data
        
        CSV should have columns:
        area_m2, depth_m, volume_liters, hotmix_kg, tack_coat_liters, aggregate_base_kg
        
        Example CSV:
        area_m2,depth_m,volume_liters,hotmix_kg,tack_coat_liters,aggregate_base_kg
        0.05,0.02,0.1,0.24,0.015,0
        0.15,0.05,0.5,1.2,0.045,0.1
        0.3,0.12,2.0,3.84,0.09,1.92
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = self.feature_columns + self.target_columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # Check for valid data
        if len(df) < 5:
            raise ValueError(f"Need at least 5 repair records, but got {len(df)}")
        
        # Handle missing values
        df = df.dropna()
        
        print(f"✓ Loaded {len(df)} repair records")
        print(f"  Feature range - area_m2: {df['area_m2'].min()}-{df['area_m2'].max()}")
        print(f"  Feature range - depth_m: {df['depth_m'].min()}-{df['depth_m'].max()}")
        print(f"  Feature range - volume_liters: {df['volume_liters'].min()}-{df['volume_liters'].max()}")
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        Train separate models for each material type
        
        Args:
            df: DataFrame with training data
            test_size: Fraction for test set (0.2 = 20%)
            random_state: For reproducibility
        """
        X = df[self.feature_columns]
        
        # Split data once for all models
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, range(len(X)), test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Train-test split: {len(X_train)} train, {len(X_test)} test")
        
        # Train separate model for each material
        for target_col in self.target_columns:
            print(f"\n--- Training model for {target_col} ---")
            y_train = df.loc[indices_train, target_col].values
            y_test = df.loc[indices_test, target_col].values
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train model
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:  # gradient_boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=random_state
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                       scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            print(f"  MAE:      {mae:.4f}")
            print(f"  RMSE:     {rmse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  CV MAE (5-fold): {cv_mae:.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                print(f"  Feature Importance:")
                for feat, imp in zip(self.feature_columns, model.feature_importances_):
                    print(f"    {feat}: {imp:.3f}")
            
            self.models[target_col] = model
            self.scalers[target_col] = scaler
    
    def predict(self, area_m2, depth_m, volume_liters):
        """
        Predict material quantities for a new pothole
        
        Args:
            area_m2: Surface area in square meters
            depth_m: Depth in meters
            volume_liters: Volume in liters
        
        Returns:
            dict with predicted hotmix_kg, tack_coat_liters, aggregate_base_kg
        """
        X = np.array([[area_m2, depth_m, volume_liters]])
        
        predictions = {}
        for target_col in self.target_columns:
            scaler = self.scalers[target_col]
            model = self.models[target_col]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            # Ensure non-negative predictions
            predictions[target_col] = max(0, pred)
        
        return {
            "hotmix_kg": round(predictions['hotmix_kg'], 2),
            "tack_coat_liters": round(predictions['tack_coat_liters'], 3),
            "aggregate_base_kg": round(predictions['aggregate_base_kg'], 2),
        }
    
    def save_model(self, save_dir='backend/models'):
        """
        Save trained models and scalers
        
        Args:
            save_dir: Directory to save files
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for target_col in self.target_columns:
            model_file = f"{save_dir}/material_{target_col.replace(' ', '_')}.pkl"
            scaler_file = f"{save_dir}/scaler_{target_col.replace(' ', '_')}.pkl"
            
            joblib.dump(self.models[target_col], model_file)
            joblib.dump(self.scalers[target_col], scaler_file)
            print(f"✓ Saved {model_file}")
            print(f"✓ Saved {scaler_file}")
    
    def load_model(self, load_dir='backend/models'):
        """
        Load pre-trained models and scalers
        
        Args:
            load_dir: Directory with saved models
        """
        for target_col in self.target_columns:
            model_file = f"{load_dir}/material_{target_col.replace(' ', '_')}.pkl"
            scaler_file = f"{load_dir}/scaler_{target_col.replace(' ', '_')}.pkl"
            
            try:
                self.models[target_col] = joblib.load(model_file)
                self.scalers[target_col] = joblib.load(scaler_file)
                print(f"✓ Loaded {model_file}")
            except FileNotFoundError:
                print(f"✗ Model not found: {model_file}")
                return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Train ML model for material estimation')
    parser.add_argument('--train', type=str, help='Train with CSV file (e.g., repair_data.csv)')
    parser.add_argument('--model', type=str, default='random_forest', 
                       choices=['random_forest', 'gradient_boosting'],
                       help='Model type to train')
    parser.add_argument('--save-dir', type=str, default='backend/models',
                       help='Directory to save models')
    parser.add_argument('--test-pred', action='store_true', help='Test predictions after training')
    
    args = parser.parse_args()
    
    trainer = MaterialEstimatorML(model_type=args.model)
    
    if args.train:
        print(f"🔄 Training {args.model} model on {args.train}...\n")
        df = trainer.load_data(args.train)
        trainer.train(df)
        trainer.save_model(args.save_dir)
        
        if args.test_pred:
            print("\n--- Test Predictions ---")
            test_cases = [
                (0.05, 0.02, 0.1),    # Small LOW pothole
                (0.15, 0.05, 0.5),    # Medium pothole
                (0.3, 0.1, 2.0),      # Large HIGH pothole
                (0.5, 0.15, 3.0),     # Critical pothole
            ]
            for area, depth, volume in test_cases:
                pred = trainer.predict(area, depth, volume)
                print(f"Area={area}m², Depth={depth}m, Volume={volume}L")
                print(f"  → HMA: {pred['hotmix_kg']}kg, Tack: {pred['tack_coat_liters']}L, Aggregate: {pred['aggregate_base_kg']}kg\n")


if __name__ == '__main__':
    main()
