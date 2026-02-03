import json
import os
from sqlalchemy import text
from db import SessionLocal, engine
from models import Base

def create_tables_only():
    print("Creating tables...")
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.commit()
    except Exception as e:
        print(f"Warning creating extension: {e}")
        return {"status": "warning", "message": f"Extension warning: {e}"}

    try:
        Base.metadata.create_all(bind=engine)
        return {"status": "success", "message": "Tables created successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def import_data_only():
    print("Loading GeoJSON...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data/roads.geojson")
    
    try:
        with open(json_path, encoding="utf-8") as f:
            geo = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": f"{json_path} not found!"}

    db = SessionLocal()
    count = 0
    try:
        print(f"Importing {len(geo['features'])} roads...")
        for f in geo["features"]:
            p = f["properties"]
            g = json.dumps(f["geometry"])
            db.execute(
                text("""
                    INSERT INTO roads
                    (road_id, road_name, city, road_weight, geom)
                    VALUES
                    (:road_id, :road_name, :city, :road_weight,
                     ST_SetSRID(ST_GeomFromGeoJSON(:geom),4326))
                    ON CONFLICT (road_id) DO UPDATE SET
                    road_name = EXCLUDED.road_name,
                    geom = EXCLUDED.geom
                """),
                {
                    "road_id": p["road_id"],
                    "road_name": p["road_name"],
                    "city": p["city"],
                    "road_weight": p["road_weight"],
                    "geom": g
                }
            )
            count += 1
        db.commit()
        return {"status": "success", "imported": count}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

def init_db():
    create_res = create_tables_only()
    if create_res.get("status") == "error":
        return create_res
    
    import_res = import_data_only()
    return {
        "create_tables": create_res,
        "import_data": import_res
    }

if __name__ == "__main__":
    init_db()
