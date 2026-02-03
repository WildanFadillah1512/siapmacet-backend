from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import text
from db import SessionLocal
from cache import get_cache, set_cache
from scheduler import start_scheduler
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/roads")
def roads():
    db = SessionLocal()
    rows = db.execute(
        text("""
            SELECT road_id, road_name, road_weight, ST_AsGeoJSON(geom) AS geom
            FROM roads
        """)
    ).fetchall()
    db.close()

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "road_id": r.road_id,
                    "road_name": r.road_name,
                    "road_weight": r.road_weight
                },
                "geometry": json.loads(r.geom)
            }
            for r in rows
        ]
    }

@app.get("/traffic_snapshot")
def traffic_snapshot():
    cached = get_cache("traffic_snapshot")
    if cached:
        return json.loads(cached)

    db = SessionLocal()
    rows = db.execute(
        text("""
            SELECT road_id, speed, free_flow, confidence, created_at
            FROM traffic_history
            WHERE created_at >= NOW() - INTERVAL '10 minutes'
            ORDER BY created_at DESC
        """)
    ).fetchall()
    
    # Get last update time
    last_update_row = db.execute(
        text("SELECT MAX(created_at) as last_update FROM traffic_history")
    ).fetchone()
    db.close()

    # Deduplicate by road_id (keep most recent)
    road_data = {}
    for r in rows:
        if r.road_id not in road_data:
            road_data[r.road_id] = {
                "road_id": r.road_id,
                "speed": r.speed,
                "free_flow": r.free_flow,
                "confidence": r.confidence
            }
    
    result = list(road_data.values())
    set_cache("traffic_snapshot", json.dumps(result), 60)
    return result


@app.get("/system_status")
def system_status():
    """
    Get real-time system status including last update time and API health.
    """
    from datetime import datetime, timezone
    
    db = SessionLocal()
    
    # Last traffic update
    last_update_row = db.execute(
        text("SELECT MAX(created_at) as last_update FROM traffic_history")
    ).fetchone()
    
    # Count recent updates (last hour)
    recent_count = db.execute(
        text("SELECT COUNT(*) as cnt FROM traffic_history WHERE created_at >= NOW() - INTERVAL '1 hour'")
    ).fetchone()
    
    db.close()
    
    last_update = last_update_row.last_update if last_update_row else None
    
    # Calculate seconds since last update
    if last_update:
        now = datetime.now(timezone.utc)
        last_update_utc = last_update.replace(tzinfo=timezone.utc) if last_update.tzinfo is None else last_update
        seconds_ago = (now - last_update_utc).total_seconds()
    else:
        seconds_ago = None
    
    # Determine status
    if seconds_ago is None:
        status = "no_data"
        status_label = "No Data"
    elif seconds_ago < 300:  # 5 minutes
        status = "live"
        status_label = "Live"
    elif seconds_ago < 600:  # 10 minutes
        status = "delayed"
        status_label = "Delayed"
    else:
        status = "offline"
        status_label = "Offline"
    
    # Get API key status
    try:
        from tomtom import get_api_status
        api_status = get_api_status()
    except:
        api_status = {"error": "unavailable"}
    
    return {
        "status": status,
        "status_label": status_label,
        "last_update": last_update.isoformat() if last_update else None,
        "seconds_ago": round(seconds_ago) if seconds_ago else None,
        "updates_last_hour": recent_count.cnt if recent_count else 0,
        "api_keys": api_status
    }


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/analytics/hourly/{road_id}")
def analytics_hourly(road_id: str):
    """
    Get 24h hourly speed trend for a road.
    """
    try:
        from analytics import get_hourly_trend
        return get_hourly_trend(road_id)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/analytics/heatmap")
def analytics_heatmap():
    """
    Get heatmap data (lat, lon, intensity) of congestion in last 24h.
    """
    try:
        from analytics import get_heatmap_data
        data = get_heatmap_data()
        return {"points": data}  # [[lat, lon, intensity], ...]
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# ML ENDPOINTS
# ============================================================================

# Lazy-loaded model cache
_ml_model_cache = {}


def get_prediction_model():
    """Lazy load prediction model."""
    import os
    if 'prediction' not in _ml_model_cache:
        model_path = os.path.join(os.path.dirname(__file__), "models", "prediction_model.pkl")
        if os.path.exists(model_path):
            import joblib
            _ml_model_cache['prediction'] = joblib.load(model_path)
        else:
            return None
    return _ml_model_cache['prediction']


@app.get("/road_clusters")
def get_road_clusters():
    """
    Get cluster assignments for all roads.
    
    Returns list of roads with their cluster_id and label.
    Clustering groups roads by traffic pattern, not location.
    """
    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT 
                    rc.road_id, 
                    rc.cluster_id, 
                    rc.cluster_label,
                    rc.cluster_description,
                    r.road_name
                FROM road_clusters rc
                JOIN roads r ON rc.road_id = r.road_id
                ORDER BY rc.cluster_id, rc.road_id
            """)
        ).fetchall()
        
        if not rows:
            return {"error": "No cluster data. Run 'python -m ml.train' first."}
        
        return [
            {
                "road_id": r.road_id,
                "road_name": r.road_name,
                "cluster_id": r.cluster_id,
                "cluster_label": r.cluster_label,
                "cluster_description": r.cluster_description
            }
            for r in rows
        ]
    finally:
        db.close()


@app.get("/forecast_ml")
def forecast_ml(road_id: str):
    """
    Predict traffic condition 30 minutes ahead for a specific road.
    
    Uses XGBoost model trained on historical traffic patterns.
    
    Args:
        road_id: Road ID to predict for (e.g., SBM_SKC_01)
        
    Returns:
        Prediction with current status, forecast, and confidence.
    """
    model_data = get_prediction_model()
    
    if model_data is None:
        return {"error": "Model not trained. Run 'python -m ml.train' first."}
    
    try:
        from ml.prediction import predict_for_road
        return predict_for_road(road_id, model_data)
    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast_ml_all")
def forecast_ml_all():
    """
    Predict traffic for all roads with available data.
    
    Returns list of predictions for each road.
    Cached for 60 seconds.
    """
    cached = get_cache("forecast_ml_all")
    if cached:
        return json.loads(cached)
    
    model_data = get_prediction_model()
    
    if model_data is None:
        return {"error": "Model not trained. Run 'python -m ml.train' first."}
    
    try:
        from ml.prediction import predict_all_roads
        result = predict_all_roads(model_data)
        set_cache("forecast_ml_all", json.dumps(result), 60)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def get_forecast():
    """
    Get traffic forecast with phased ML architecture.
    
    Phase 1 (Data Collection): Returns estimation-based forecast with confidence %
    Phase 2 (30+ days data): Returns supervised learning predictions
    
    Auto-switches when sufficient data is available.
    """
    cached = get_cache("forecast_phased")
    if cached:
        return json.loads(cached)
    
    try:
        # Get current traffic data
        traffic_data = traffic_snapshot()
        
        # Get forecast from phased module
        from ml.forecast import get_forecast_for_roads
        result = get_forecast_for_roads(traffic_data)
        
        set_cache("forecast_phased", json.dumps(result), 60)
        return result
    except Exception as e:
        return {"error": str(e), "method": "fallback"}


@app.get("/data_status")
def get_data_status():
    """
    Get status of data collection for ML training.
    
    Returns:
        - Data collection progress percentage
        - Days of data collected
        - Days remaining until ML is ready
        - Whether supervised learning is available
    """
    try:
        from ml.forecast import get_data_stats
        return get_data_stats()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api_key_status")
def api_key_status():
    """
    Get status of TomTom API key usage and rotation.
    
    Returns:
        - Total keys configured
        - Current active key
        - Usage per key
        - Remaining capacity
    """
    try:
        from tomtom import get_api_status
        return get_api_status()
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug_init_db")
def debug_init_db():
    from loader_geojson import init_db
    result = init_db()
    return result
