import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    use_native_hstore=False
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
