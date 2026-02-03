from sqlalchemy import text
from db import SessionLocal
import json

def get_hourly_trend(road_id: str):
    """
    Get average speed per hour for the last 24 hours for a specific road.
    """
    db = SessionLocal()
    try:
        # Postgres query to bucket by hour
        result = db.execute(
            text("""
                SELECT 
                    to_char(created_at, 'HH24:00') as time_label,
                    AVG(speed) as avg_speed,
                    MIN(speed) as min_speed,
                    MAX(speed) as max_speed
                FROM traffic_history
                WHERE road_id = :road_id
                  AND created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY to_char(created_at, 'HH24:00'), EXTRACT(HOUR FROM created_at)
                ORDER BY EXTRACT(HOUR FROM created_at)
            """),
            {"road_id": road_id}
        ).fetchall()
        
        return [
            {
                "time_label": r.time_label,
                "avg_speed": round(r.avg_speed, 1) if r.avg_speed else 0,
                "min_speed": r.min_speed,
                "max_speed": r.max_speed
            }
            for r in result
        ]
    finally:
        db.close()

def get_heatmap_data():
    """
    Get high congestion points (speed < 20 km/h) from the last 24 hours.
    Returns list of [lat, lon, intensity].
    Intensity is based on frequency of congestion at that point.
    """
    db = SessionLocal()
    try:
        # Join traffic_history with roads to get coordinates
        # We want locations where traffic was heavy (speed < 20)
        # We aggregate by road to get a heat intensity (count of heavy traffic records)
        result = db.execute(
            text("""
                SELECT 
                    ST_Y(ST_Centroid(r.geom)) as lat,
                    ST_X(ST_Centroid(r.geom)) as lon,
                    COUNT(*) as congestion_count
                FROM traffic_history th
                JOIN roads r ON th.road_id = r.road_id
                WHERE th.speed < 20
                  AND th.created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY r.road_id, r.geom
            """)
        ).fetchall()
        
        # Normalize intensity 0-1 (optional, or just return raw count and handle in frontend)
        # Let's return raw count for now, Leaflet heat uses this.
        
        data = []
        max_count = 0
        for r in result:
            count = r.congestion_count
            if count > max_count:
                max_count = count
            data.append([r.lat, r.lon, count])
            
        # Normalize intensity to 0.0 - 1.0 range if max_count > 0
        normalized_data = []
        if max_count > 0:
            for item in data:
                # [lat, lon, normalized_intensity]
                # Intensity factor: more frequent congestion = hotter
                normalized_data.append([item[0], item[1], round(item[2] / max_count, 2)])
                
        return normalized_data
    finally:
        db.close()
