"""
Batch Training Script for SiapMacet ML Pipeline.

Run this script to train all ML models:
    python -m ml.train

This script:
1. Runs clustering to group roads by traffic patterns
2. Trains prediction model using cluster_id as feature
3. Saves models to disk

Schedule this script to run periodically (daily/weekly) via cron or scheduler.
"""

import os
import sys
from datetime import datetime

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.clustering import run_clustering, ensure_road_clusters_table
from ml.prediction import train_prediction_model


def train_all_models(verbose: bool = True):
    """
    Train all ML models in the correct order.
    
    Order matters:
    1. Clustering first (creates cluster_id)
    2. Prediction second (uses cluster_id as feature)
    """
    start_time = datetime.now()
    
    if verbose:
        print("=" * 70)
        print("SIAPMACET ML TRAINING PIPELINE")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    results = {}
    
    # Step 1: Clustering
    try:
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1: CLUSTERING")
            print("=" * 70)
        
        cluster_df = run_clustering(n_clusters=4, save_to_db=True)
        results['clustering'] = {
            'status': 'success',
            'n_roads': len(cluster_df),
            'n_clusters': cluster_df['cluster_id'].nunique()
        }
    except Exception as e:
        if verbose:
            print(f"\nClustering failed: {e}")
        results['clustering'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Step 2: Prediction Model
    try:
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: PREDICTION MODEL")
            print("=" * 70)
        
        model, metrics = train_prediction_model(save_model=True)
        results['prediction'] = {
            'status': 'success',
            'metrics': metrics
        }
    except Exception as e:
        if verbose:
            print(f"\nPrediction training failed: {e}")
        results['prediction'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nDuration: {duration:.1f} seconds")
        print("\nResults Summary:")
        
        for step, result in results.items():
            status = result.get('status', 'unknown')
            icon = "✓" if status == 'success' else "✗"
            print(f"  {icon} {step}: {status}")
            
            if status == 'success':
                if 'metrics' in result:
                    print(f"      Accuracy: {result['metrics'].get('accuracy', 'N/A'):.4f}")
                    print(f"      F1 Score: {result['metrics'].get('f1_weighted', 'N/A'):.4f}")
                if 'n_roads' in result:
                    print(f"      Roads clustered: {result['n_roads']}")
    
    return results


if __name__ == "__main__":
    # Handle command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SiapMacet ML models')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output')
    parser.add_argument('--clustering-only', action='store_true', help='Only run clustering')
    parser.add_argument('--prediction-only', action='store_true', help='Only run prediction')
    
    args = parser.parse_args()
    
    if args.clustering_only:
        run_clustering(n_clusters=4, save_to_db=True)
    elif args.prediction_only:
        train_prediction_model(save_model=True)
    else:
        train_all_models(verbose=not args.quiet)
