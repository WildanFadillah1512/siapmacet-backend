from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text
from db import SessionLocal
from tomtom import fetch_traffic

scheduler = BackgroundScheduler()

def update_traffic():
    db = SessionLocal()
    roads = db.execute(
        text("""
            SELECT road_id,
                   ST_Y(ST_Centroid(geom)) AS lat,
                   ST_X(ST_Centroid(geom)) AS lon
            FROM roads
        """)
    ).fetchall()

    for r in roads:
        data = fetch_traffic(r.lat, r.lon)
        if not data:
            continue

        db.execute(
            text("""
                INSERT INTO traffic_history
                (road_id, speed, free_flow, confidence)
                VALUES (:road_id, :speed, :free_flow, :confidence)
            """),
            {
                "road_id": r.road_id,
                "speed": data["speed"],
                "free_flow": data["free_flow"],
                "confidence": data["confidence"]
            }
        )

    db.commit()
    db.close()

def start_scheduler():
    scheduler.add_job(update_traffic, "interval", minutes=5)
    scheduler.start()
