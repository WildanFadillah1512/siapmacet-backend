"""
Feature Engineering Module for SiapMacet ML Pipeline.

This module provides functions to extract features from traffic_history
for both clustering and prediction tasks.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Import database connection
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal


# ============================================================================
# SQL-BASED FEATURE EXTRACTION
# ============================================================================

TRAFFIC_FEATURES_SQL = """
SELECT
    th.id,
    th.road_id,
    th.speed,
    th.free_flow,
    th.confidence,
    th.created_at,
    
    -- Temporal Features
    EXTRACT(HOUR FROM th.created_at)::INTEGER AS hour_of_day,
    EXTRACT(DOW FROM th.created_at)::INTEGER AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM th.created_at) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN EXTRACT(HOUR FROM th.created_at) BETWEEN 6 AND 9 
         OR EXTRACT(HOUR FROM th.created_at) BETWEEN 16 AND 19 
         THEN 1 ELSE 0 END AS is_peak_hour,
    
    -- Traffic Features
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
    
    -- Spatial Feature
    r.road_weight
    
FROM traffic_history th
LEFT JOIN roads r ON th.road_id = r.road_id
WHERE th.speed IS NOT NULL
ORDER BY th.road_id, th.created_at
"""


def fetch_traffic_features() -> pd.DataFrame:
    """
    Fetch traffic data with engineered features from database.
    
    Returns:
        DataFrame with temporal, traffic, and spatial features.
    """
    db = SessionLocal()
    try:
        result = db.execute(text(TRAFFIC_FEATURES_SQL))
        rows = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
        return df
    finally:
        db.close()


# ============================================================================
# ROLLING WINDOW FEATURES (Time-Series)
# ============================================================================

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series rolling window features per road_id.
    
    Features added:
    - rolling_mean_15min: 3 records * 5min = 15min rolling mean
    - rolling_mean_30min: 6 records * 5min = 30min rolling mean  
    - rolling_std_30min: 30min rolling standard deviation
    
    Args:
        df: DataFrame with at least 'road_id', 'speed', 'created_at' columns
        
    Returns:
        DataFrame with additional rolling features
    """
    df = df.copy()
    df = df.sort_values(['road_id', 'created_at'])
    
    # Group by road_id and calculate rolling features
    rolling_features = []
    
    for road_id in df['road_id'].unique():
        mask = df['road_id'] == road_id
        road_df = df.loc[mask].copy()
        
        # Rolling mean 15 minutes (3 * 5min intervals)
        road_df['rolling_mean_15min'] = (
            road_df['speed']
            .rolling(window=3, min_periods=1)
            .mean()
        )
        
        # Rolling mean 30 minutes (6 * 5min intervals)
        road_df['rolling_mean_30min'] = (
            road_df['speed']
            .rolling(window=6, min_periods=1)
            .mean()
        )
        
        # Rolling std 30 minutes
        road_df['rolling_std_30min'] = (
            road_df['speed']
            .rolling(window=6, min_periods=1)
            .std()
            .fillna(0)
        )
        
        rolling_features.append(road_df)
    
    return pd.concat(rolling_features, ignore_index=True)


# ============================================================================
# FULL FEATURE PIPELINE
# ============================================================================

def get_ml_ready_dataframe() -> pd.DataFrame:
    """
    Get complete ML-ready DataFrame with all engineered features.
    
    This is the main entry point for feature engineering.
    Pipeline:
    1. Fetch base features from database (SQL)
    2. Add rolling window features (Python)
    3. Handle edge cases and missing values
    
    Returns:
        DataFrame ready for ML training/inference
    """
    # Step 1: Fetch base features
    df = fetch_traffic_features()
    
    if df.empty:
        print("Warning: No traffic data found in database")
        return df
    
    # Step 2: Add rolling features
    df = add_rolling_features(df)
    
    # Step 3: Handle edge cases
    # Fill NaN road_weight with median
    if df['road_weight'].isna().any():
        median_weight = df['road_weight'].median()
        df['road_weight'] = df['road_weight'].fillna(median_weight if not pd.isna(median_weight) else 0.5)
    
    # Ensure no NaN in critical columns
    df['speed_ratio'] = df['speed_ratio'].fillna(1.0)
    df['congestion_level'] = df['congestion_level'].fillna(0)
    
    print(f"Feature engineering complete: {len(df)} records, {len(df.columns)} features")
    return df


# ============================================================================
# ROAD-LEVEL AGGREGATION (for Clustering)
# ============================================================================

def aggregate_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate traffic features per road_id for clustering.
    
    Each road gets summarized by its typical traffic patterns.
    
    Args:
        df: ML-ready DataFrame from get_ml_ready_dataframe()
        
    Returns:
        DataFrame with one row per road_id
    """
    agg = df.groupby('road_id').agg({
        'speed': ['mean', 'std', 'min', 'max'],
        'speed_ratio': ['mean', 'std'],
        'congestion_level': ['mean', 'max'],
        'is_peak_hour': 'mean',  # proportion of records during peak
        'road_weight': 'first'
    })
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns]
    agg = agg.reset_index()
    
    # Fill NaN std with 0 (roads with only 1 record)
    for col in agg.columns:
        if 'std' in col:
            agg[col] = agg[col].fillna(0)
    
    return agg


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_names_for_clustering() -> list:
    """Return feature column names used for clustering."""
    return [
        'speed_mean',
        'speed_std', 
        'speed_ratio_mean',
        'congestion_level_mean',
        'is_peak_hour_mean',
        'road_weight_first'
    ]


def get_feature_names_for_prediction() -> list:
    """Return feature column names used for prediction."""
    return [
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        'is_peak_hour',
        'speed',
        'speed_ratio',
        'congestion_level',
        'rolling_mean_15min',
        'rolling_mean_30min',
        'rolling_std_30min',
        'road_weight'
    ]


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Feature Engineering Pipeline...")
    print("=" * 60)
    
    # Test full pipeline
    df = get_ml_ready_dataframe()
    
    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())
        
        print("\n" + "=" * 60)
        print("Testing Road Aggregation...")
        
        agg_df = aggregate_road_features(df)
        print(f"\nAggregated shape: {agg_df.shape}")
        print(f"\nSample aggregated data:")
        print(agg_df.head())
    else:
        print("No data available. Make sure traffic_history has records.")
