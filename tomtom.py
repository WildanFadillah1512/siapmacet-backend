"""
TomTom Traffic API with Multi-Key Rotation

Supports multiple API keys with automatic rotation when:
1. Daily limit (2500) is reached
2. API returns 403/429 error

Usage tracking resets at midnight.
"""

import os
import requests
from datetime import datetime, date
from dotenv import load_dotenv
import threading

load_dotenv()


class TomTomKeyManager:
    """Manages multiple TomTom API keys with rotation."""
    
    DAILY_LIMIT = 2400  # Stay under 2500 for safety margin
    
    def __init__(self):
        self.keys = self._load_keys()
        self.usage = {key: 0 for key in self.keys}
        self.last_reset_date = date.today()
        self.current_index = 0
        self.lock = threading.Lock()
        
        if not self.keys:
            print("⚠️ WARNING: No TomTom API keys found!")
            print("   Add TOMTOM_API_KEY or TOMTOM_API_KEY_1, TOMTOM_API_KEY_2, etc. to .env")
    
    def _load_keys(self) -> list:
        """Load all API keys from environment."""
        keys = []
        
        # Check single key first
        single_key = os.getenv("TOMTOM_API_KEY")
        if single_key:
            keys.append(single_key)
        
        # Check numbered keys (TOMTOM_API_KEY_1, TOMTOM_API_KEY_2, etc.)
        for i in range(1, 20):  # Support up to 20 keys
            key = os.getenv(f"TOMTOM_API_KEY_{i}")
            if key:
                keys.append(key)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        
        return unique_keys
    
    def _reset_if_new_day(self):
        """Reset usage counts at midnight."""
        today = date.today()
        if today > self.last_reset_date:
            self.usage = {key: 0 for key in self.keys}
            self.last_reset_date = today
            self.current_index = 0
            print(f"🔄 Daily reset: All API key counters reset for {today}")
    
    def get_next_key(self) -> str | None:
        """Get next available API key with usage under limit."""
        with self.lock:
            self._reset_if_new_day()
            
            if not self.keys:
                return None
            
            # Try to find a key with available quota
            attempts = 0
            while attempts < len(self.keys):
                key = self.keys[self.current_index]
                if self.usage[key] < self.DAILY_LIMIT:
                    return key
                
                # This key is exhausted, try next
                self.current_index = (self.current_index + 1) % len(self.keys)
                attempts += 1
            
            # All keys exhausted
            print("⚠️ All API keys have reached daily limit!")
            return None
    
    def record_usage(self, key: str, success: bool = True):
        """Record API usage for a key."""
        with self.lock:
            if key in self.usage:
                self.usage[key] += 1
                
                # If this key hit limit, switch to next
                if self.usage[key] >= self.DAILY_LIMIT:
                    old_index = self.current_index
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    print(f"🔄 API key #{old_index + 1} reached limit, switching to key #{self.current_index + 1}")
    
    def mark_key_failed(self, key: str):
        """Mark a key as failed (e.g., 403 error) - move to next key."""
        with self.lock:
            if key in self.usage:
                # Set to limit so it won't be used again today
                self.usage[key] = self.DAILY_LIMIT
                self.current_index = (self.current_index + 1) % len(self.keys)
                print(f"⚠️ API key marked as failed, switching to next key")
    
    def get_status(self) -> dict:
        """Get current status of all API keys."""
        self._reset_if_new_day()
        
        total_capacity = len(self.keys) * self.DAILY_LIMIT
        total_used = sum(self.usage.values())
        
        return {
            "total_keys": len(self.keys),
            "current_key_index": self.current_index + 1,
            "total_capacity": total_capacity,
            "total_used": total_used,
            "remaining": total_capacity - total_used,
            "keys_status": [
                {
                    "key_number": i + 1,
                    "used": self.usage[key],
                    "remaining": max(0, self.DAILY_LIMIT - self.usage[key]),
                    "is_current": i == self.current_index,
                    "is_exhausted": self.usage[key] >= self.DAILY_LIMIT
                }
                for i, key in enumerate(self.keys)
            ]
        }


# Global key manager instance
_key_manager = None


def get_key_manager() -> TomTomKeyManager:
    """Get or create the global key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = TomTomKeyManager()
    return _key_manager


def fetch_traffic(lat: float, lon: float) -> dict | None:
    """
    Fetch traffic data from TomTom API with automatic key rotation.
    
    Returns:
        dict with speed, free_flow, confidence or None if failed
    """
    manager = get_key_manager()
    
    # Try up to 3 times with different keys
    for attempt in range(3):
        api_key = manager.get_next_key()
        
        if api_key is None:
            print("❌ No available API keys!")
            return None
        
        try:
            url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {"key": api_key, "point": f"{lat},{lon}"}
            
            r = requests.get(url, params=params, timeout=10)
            
            if r.status_code == 200:
                manager.record_usage(api_key, success=True)
                d = r.json()["flowSegmentData"]
                return {
                    "speed": d["currentSpeed"],
                    "free_flow": d["freeFlowSpeed"],
                    "confidence": d["confidence"]
                }
            elif r.status_code in [403, 429]:
                # Rate limited or forbidden - try next key
                print(f"⚠️ API key got {r.status_code}, rotating...")
                manager.mark_key_failed(api_key)
                continue
            else:
                manager.record_usage(api_key, success=False)
                print(f"❌ TomTom API error: {r.status_code}")
                return None
                
        except requests.Timeout:
            manager.record_usage(api_key, success=False)
            print(f"⚠️ TomTom API timeout")
            return None
        except Exception as e:
            manager.record_usage(api_key, success=False)
            print(f"❌ TomTom API exception: {e}")
            return None
    
    return None


def get_api_status() -> dict:
    """Get status of API key usage."""
    return get_key_manager().get_status()
