from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()

class Road(Base):
    __tablename__ = "roads"
    id = Column(Integer, primary_key=True)
    road_id = Column(String, unique=True, nullable=False)
    road_name = Column(String)
    city = Column(String)
    road_weight = Column(Float)
    geom = Column(Geometry("LINESTRING", srid=4326))

class TrafficHistory(Base):
    __tablename__ = "traffic_history"
    id = Column(Integer, primary_key=True)
    road_id = Column(String, nullable=False)  # ForeignKey if needed, but loose coupling is fine
    speed = Column(Integer)
    free_flow = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

