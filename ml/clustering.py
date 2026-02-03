"""
Clustering Module for SiapMacet ML Pipeline.

This module implements unsupervised learning to group roads
based on their traffic patterns, not just location.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Import from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal
from ml.feature_engineering import get_ml_ready_dataframe, aggregate_road_features, get_feature_names_for_clustering


# ============================================================================
# CLUSTER LABELS AND INTERPRETATION
# ============================================================================

CLUSTER_LABELS = {
    0: "stable_smooth",      # Jalan stabil, jarang macet
    1: "peak_congested",     # Macet saat jam sibuk
    2: "frequently_jammed",  # Sering macet sepanjang hari
    3: "variable_traffic"    # Pola tidak menentu
}

CLUSTER_DESCRIPTIONS = {
    0: "Jalan dengan lalu lintas stabil dan lancar sepanjang hari",
    1: "Jalan yang cenderung macet saat jam sibuk (pagi/sore)",
    2: "Jalan yang sering mengalami kemacetan sepanjang hari",
    3: "Jalan dengan pola lalu lintas yang bervariasi dan tidak menentu"
}


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

CREATE_ROAD_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS road_clusters (
    road_id VARCHAR PRIMARY KEY REFERENCES roads(road_id),
    cluster_id INTEGER NOT NULL,
    cluster_label VARCHAR(50),
    cluster_description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

def ensure_road_clusters_table():
    """Create road_clusters table if not exists."""
    db = SessionLocal()
    try:
        db.execute(text(CREATE_ROAD_CLUSTERS_TABLE))
        db.commit()
        print("Table 'road_clusters' ready")
    finally:
        db.close()


def save_clusters_to_db(cluster_df: pd.DataFrame):
    """
    Save clustering results to database.
    
    Args:
        cluster_df: DataFrame with 'road_id', 'cluster_id' columns
    """
    db = SessionLocal()
    try:
        # Clear existing clusters
        db.execute(text("DELETE FROM road_clusters"))
        
        # Insert new clusters
        for _, row in cluster_df.iterrows():
            cluster_id = int(row['cluster_id'])
            db.execute(
                text("""
                    INSERT INTO road_clusters (road_id, cluster_id, cluster_label, cluster_description)
                    VALUES (:road_id, :cluster_id, :label, :description)
                    ON CONFLICT (road_id) DO UPDATE SET
                        cluster_id = :cluster_id,
                        cluster_label = :label,
                        cluster_description = :description,
                        updated_at = NOW()
                """),
                {
                    "road_id": row['road_id'],
                    "cluster_id": cluster_id,
                    "label": CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}"),
                    "description": CLUSTER_DESCRIPTIONS.get(cluster_id, "")
                }
            )
        
        db.commit()
        print(f"Saved {len(cluster_df)} road clusters to database")
    finally:
        db.close()


def load_clusters_from_db() -> pd.DataFrame:
    """Load cluster assignments from database."""
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT road_id, cluster_id, cluster_label 
            FROM road_clusters
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=['road_id', 'cluster_id', 'cluster_label'])
    finally:
        db.close()


# ============================================================================
# CLUSTERING PIPELINE
# ============================================================================

def train_clustering_model(
    n_clusters: int = 4,
    save_model: bool = True,
    model_path: str = "models/cluster_model.pkl"
) -> tuple:
    """
    Train KMeans clustering model on road-level aggregated features.
    
    Args:
        n_clusters: Number of clusters (default: 4)
        save_model: Whether to save model to disk
        model_path: Path to save model
        
    Returns:
        Tuple of (cluster_df, model, scaler)
    """
    print("=" * 60)
    print("CLUSTERING PIPELINE")
    print("=" * 60)
    
    # Step 1: Get ML-ready data
    print("\n[1/5] Fetching traffic features...")
    df = get_ml_ready_dataframe()
    
    if df.empty:
        raise ValueError("No traffic data available for clustering")
    
    # Step 2: Aggregate per road
    print("[2/5] Aggregating features per road...")
    agg_df = aggregate_road_features(df)
    print(f"      Found {len(agg_df)} roads")
    
    # Step 3: Prepare features for clustering
    print("[3/5] Preparing feature matrix...")
    feature_cols = get_feature_names_for_clustering()
    X = agg_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Fit KMeans
    print(f"[4/5] Training KMeans with {n_clusters} clusters...")
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    cluster_labels = model.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    agg_df['cluster_id'] = cluster_labels
    
    # Step 5: Analyze clusters
    print("[5/5] Analyzing cluster characteristics...")
    analyze_clusters(agg_df, feature_cols)
    
    # Save model if requested
    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    return agg_df[['road_id', 'cluster_id']], model, scaler


def analyze_clusters(df: pd.DataFrame, feature_cols: list):
    """Print cluster statistics and characteristics."""
    print("\n" + "-" * 40)
    print("CLUSTER ANALYSIS")
    print("-" * 40)
    
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_data = df[df['cluster_id'] == cluster_id]
        label = CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}")
        
        print(f"\n[Cluster {cluster_id}] {label}")
        print(f"  Roads: {len(cluster_data)}")
        
        for col in feature_cols:
            mean_val = cluster_data[col].mean()
            print(f"  {col}: {mean_val:.3f}")


def assign_cluster_labels_by_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically assign cluster labels based on actual cluster characteristics.
    
    This ensures labels match the actual traffic patterns in each cluster.
    """
    cluster_stats = df.groupby('cluster_id').agg({
        'speed_mean': 'mean',
        'speed_std': 'mean',
        'congestion_level_mean': 'mean',
        'is_peak_hour_mean': 'mean'
    })
    
    # Sort clusters by congestion level
    cluster_order = cluster_stats.sort_values('congestion_level_mean').index.tolist()
    
    # Map to meaningful labels
    label_map = {}
    n_clusters = len(cluster_order)
    
    if n_clusters >= 4:
        label_map[cluster_order[0]] = 0  # stable_smooth (lowest congestion)
        label_map[cluster_order[-1]] = 2  # frequently_jammed (highest congestion)
        
        # Determine peak_congested vs variable based on peak_hour correlation
        remaining = cluster_order[1:-1]
        for i, c in enumerate(remaining):
            if cluster_stats.loc[c, 'is_peak_hour_mean'] > 0.4:
                label_map[c] = 1  # peak_congested
            else:
                label_map[c] = 3  # variable_traffic
    else:
        # Fallback for fewer clusters
        for i, c in enumerate(cluster_order):
            label_map[c] = i
    
    df['cluster_id_mapped'] = df['cluster_id'].map(label_map)
    return df


# ============================================================================
# MAIN RUN FUNCTION
# ============================================================================

def run_clustering(n_clusters: int = 4, save_to_db: bool = True):
    """
    Main entry point to run the complete clustering pipeline.
    
    Args:
        n_clusters: Number of clusters
        save_to_db: Whether to save results to database
    """
    # Ensure table exists
    ensure_road_clusters_table()
    
    # Train model and get clusters
    cluster_df, model, scaler = train_clustering_model(n_clusters=n_clusters)
    
    # Save to database
    if save_to_db:
        save_clusters_to_db(cluster_df)
    
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\nCluster Distribution:")
    print(cluster_df['cluster_id'].value_counts().sort_index())
    
    return cluster_df


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    print("Running Clustering Pipeline...")
    run_clustering(n_clusters=4, save_to_db=True)
