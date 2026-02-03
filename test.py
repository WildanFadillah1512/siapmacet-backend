from db import engine

conn = engine.connect()
print("DB CONNECTED")
conn.close()
