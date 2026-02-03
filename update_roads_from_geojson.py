import json
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://postgres:Reville091@localhost:5432/siap_macet"
)

with open(r"C:\SiapMacet\backend\map.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

with engine.begin() as conn:
    for feature in data["features"]:
        props = feature["properties"]

        conn.execute(
            text("""
                UPDATE roads
                SET
                    road_name = :road_name,
                    city = :city,
                    road_weight = :road_weight
                WHERE road_id = :road_id
            """),
            {
                "road_id": props["road_id"],
                "road_name": props["road_name"],
                "city": props["city"],
                "road_weight": props["road_weight"]
            }
        )

print("UPDATE roads selesai.")
