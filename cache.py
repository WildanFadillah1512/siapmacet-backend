import time
import redis

import os
try:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    r.ping()
except:
    r = None

_memory_cache = {}

def get_cache(key):
    if r:
        return r.get(key)
    v = _memory_cache.get(key)
    if not v:
        return None
    value, exp = v
    if time.time() > exp:
        del _memory_cache[key]
        return None
    return value

def set_cache(key, value, ttl=60):
    if r:
        r.setex(key, ttl, value)
    else:
        _memory_cache[key] = (value, time.time() + ttl)
