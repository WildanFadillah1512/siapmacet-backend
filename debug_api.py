from tomtom import get_key_manager
import requests
import json

manager = get_key_manager()
print(f"🔑 Total keys found: {len(manager.keys)}")

# Get current key
key = manager.get_next_key()
print(f"👉 Testing with key: {key}")

if key:
    # Sukabumi coordinates
    lat, lon = -6.9216, 106.9239
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"key": key, "point": f"{lat},{lon}"}
    
    print(f"📡 Requesting: {url}")
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"📝 Status Code: {r.status_code}")
        print(f"📄 Response: {r.text[:500]}")
    except Exception as e:
        print(f"❌ Exception: {e}")
else:
    print("❌ No keys available!")
