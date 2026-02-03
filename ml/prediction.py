"""
Prediction Module for SiapMacet ML Pipeline.

This module implements supervised learning to predict traffic conditions
30 minutes ahead using XGBoost classifier.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Import from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal
from ml.feature_engineering import get_ml_ready_dataframe, get_feature_names_for_prediction
from ml.clustering import load_clusters_from_db


# ============================================================================
# CONSTANTS
# ============================================================================

PREDICTION_HORIZON = 6  # 6 * 5min = 30 minutes ahead
CONGESTION_LABELS = {0: "lancar", 1: "padat", 2: "macet"}
MODEL_PATH = "models/prediction_model.pkl"


# ============================================================================
# DATASET CREATION
# ============================================================================

def create_supervised_dataset(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON
) -> tuple:
    """
    Create sliding window dataset for supervised learning.
    
    For each record, we use current features to predict congestion_level
    `horizon` time steps (records) in the future.
    
    Args:
        df: ML-ready DataFrame with all features
        horizon: Number of time steps ahead to predict (default: 6 = 30min)
        
    Returns:
        Tuple of (X, y, road_ids) arrays
    """
    feature_cols = get_feature_names_for_prediction()
    
    # Add cluster_id as feature if available
    cluster_df = load_clusters_from_db()
    if not cluster_df.empty:
        df = df.merge(cluster_df[['road_id', 'cluster_id']], on='road_id', how='left')
        df['cluster_id'] = df['cluster_id'].fillna(-1)
        feature_cols = feature_cols + ['cluster_id']
    
    X, y, road_ids = [], [], []
    
    for road_id in df['road_id'].unique():
        road_df = df[df['road_id'] == road_id].sort_values('created_at').reset_index(drop=True)
        
        # Need at least horizon+1 records to create one sample
        if len(road_df) <= horizon:
            continue
        
        for i in range(len(road_df) - horizon):
            # Current features (input)
            features = road_df.loc[i, feature_cols].values.astype(float)
            
            # Future congestion level (target)
            target = int(road_df.loc[i + horizon, 'congestion_level'])
            
            X.append(features)
            y.append(target)
            road_ids.append(road_id)
    
    return np.array(X), np.array(y), np.array(road_ids), feature_cols


def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    road_ids: np.ndarray,
    train_ratio: float = 0.8
) -> tuple:
    """
    Split dataset by time, NOT randomly.
    
    This is crucial for time-series data to avoid data leakage.
    
    Args:
        X: Feature matrix
        y: Target vector
        road_ids: Road ID for each sample
        train_ratio: Proportion of data for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_prediction_model(
    save_model: bool = True,
    model_path: str = MODEL_PATH
) -> tuple:
    """
    Train XGBoost classifier for traffic prediction.
    
    Args:
        save_model: Whether to save model to disk
        model_path: Path to save model
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    print("=" * 60)
    print("PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Get ML-ready data
    print("\n[1/6] Fetching traffic features...")
    df = get_ml_ready_dataframe()
    
    if df.empty:
        raise ValueError("No traffic data available for training")
    
    print(f"      Total records: {len(df)}")
    
    # Step 2: Create supervised dataset
    print(f"[2/6] Creating sliding window dataset (horizon={PREDICTION_HORIZON})...")
    X, y, road_ids, feature_cols = create_supervised_dataset(df)
    
    if len(X) == 0:
        raise ValueError("Not enough data to create training samples")
    
    print(f"      Training samples: {len(X)}")
    print(f"      Features: {len(feature_cols)}")
    
    # Step 3: Time-based split
    print("[3/6] Splitting data (80/20 time-based)...")
    X_train, X_test, y_train, y_test = time_based_split(X, y, road_ids)
    print(f"      Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 4: Class distribution
    print("[4/6] Checking class distribution...")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = CONGESTION_LABELS.get(u, str(u))
        print(f"      {label}: {c} ({100*c/len(y_train):.1f}%)")
    
    # Step 5: Train XGBoost
    print("[5/6] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate
    print("[6/6] Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print("\n" + "-" * 40)
    print("MODEL EVALUATION")
    print("-" * 40)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    
    print("\nClassification Report:")
    target_names = [CONGESTION_LABELS[i] for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance
    print("\nTop 5 Feature Importances:")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_cols[idx]}: {importance[idx]:.4f}")
    
    # Save model
    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'horizon': PREDICTION_HORIZON,
            'trained_at': datetime.now().isoformat()
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, metrics


# ============================================================================
# PREDICTION (INFERENCE)
# ============================================================================

def load_prediction_model(model_path: str = MODEL_PATH) -> dict:
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_for_road(
    road_id: str,
    model_data: dict = None,
    model_path: str = MODEL_PATH
) -> dict:
    """
    Predict traffic condition 30 minutes ahead for a specific road.
    
    Args:
        road_id: Road ID to predict for
        model_data: Pre-loaded model data (optional)
        model_path: Path to model file
        
    Returns:
        Dictionary with prediction result
    """
    # Load model if not provided
    if model_data is None:
        model_data = load_prediction_model(model_path)
    
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # Get latest features for road
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT
                th.speed,
                th.free_flow,
                th.created_at,
                EXTRACT(HOUR FROM th.created_at)::INTEGER AS hour_of_day,
                EXTRACT(DOW FROM th.created_at)::INTEGER AS day_of_week,
                CASE WHEN EXTRACT(DOW FROM th.created_at) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
                CASE WHEN EXTRACT(HOUR FROM th.created_at) BETWEEN 6 AND 9 
                     OR EXTRACT(HOUR FROM th.created_at) BETWEEN 16 AND 19 
                     THEN 1 ELSE 0 END AS is_peak_hour,
                CASE 
                    WHEN th.free_flow = 0 OR th.free_flow IS NULL THEN 1.0
                    ELSE th.speed / th.free_flow 
                END AS speed_ratio,
                CASE
                    WHEN th.free_flow = 0 THEN 0
                    WHEN th.speed >= 35 THEN 0
                    WHEN th.speed >= 20 THEN 1
                    ELSE 2
                END AS congestion_level,
                r.road_weight,
                COALESCE(rc.cluster_id, -1) AS cluster_id
            FROM traffic_history th
            LEFT JOIN roads r ON th.road_id = r.road_id
            LEFT JOIN road_clusters rc ON th.road_id = rc.road_id
            WHERE th.road_id = :road_id
            ORDER BY th.created_at DESC
            LIMIT 6
        """), {"road_id": road_id})
        
        rows = result.fetchall()
        
        if not rows:
            return {"error": f"No data found for road_id: {road_id}"}
        
        # Calculate rolling features from recent records
        speeds = [float(r.speed) for r in rows]
        latest = rows[0]
        
        rolling_mean_15min = np.mean(speeds[:3]) if len(speeds) >= 3 else np.mean(speeds)
        rolling_mean_30min = np.mean(speeds[:6]) if len(speeds) >= 6 else np.mean(speeds)
        rolling_std_30min = np.std(speeds[:6]) if len(speeds) >= 6 else 0.0
        
        # Build feature vector
        features = {
            'hour_of_day': latest.hour_of_day,
            'day_of_week': latest.day_of_week,
            'is_weekend': latest.is_weekend,
            'is_peak_hour': latest.is_peak_hour,
            'speed': latest.speed,
            'speed_ratio': latest.speed_ratio,
            'congestion_level': latest.congestion_level,
            'rolling_mean_15min': rolling_mean_15min,
            'rolling_mean_30min': rolling_mean_30min,
            'rolling_std_30min': rolling_std_30min,
            'road_weight': latest.road_weight or 0.5,
        }
        
        # Add cluster_id if model was trained with it
        if 'cluster_id' in feature_cols:
            features['cluster_id'] = latest.cluster_id
        
        # Create feature array in correct order
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        
        # Predict
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[prediction])
        
        return {
            "road_id": road_id,
            "current_speed": float(latest.speed),
            "current_status": CONGESTION_LABELS.get(int(latest.congestion_level), "unknown"),
            "forecast_30min": CONGESTION_LABELS.get(prediction, "unknown"),
            "confidence": round(confidence, 3),
            "probabilities": {
                CONGESTION_LABELS[i]: round(float(p), 3) 
                for i, p in enumerate(probabilities)
            },
            "data_timestamp": latest.created_at.isoformat()
        }
        
    finally:
        db.close()


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_all_roads(model_data: dict = None) -> list:
    """
    Predict traffic for all roads with available data.
    
    Returns:
        List of prediction dictionaries
    """
    if model_data is None:
        model_data = load_prediction_model()
    
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT DISTINCT road_id FROM traffic_history"))
        road_ids = [r[0] for r in result.fetchall()]
    finally:
        db.close()
    
    predictions = []
    for road_id in road_ids:
        pred = predict_for_road(road_id, model_data)
        if "error" not in pred:
            predictions.append(pred)
    
    return predictions


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    print("Running Prediction Model Training...")
    
    try:
        model, metrics = train_prediction_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nFinal Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Make sure you have enough traffic data in the database.")
