from scheduler import update_traffic
import logging

# Configure logging to see what's happening
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

print("🚀 Starting manual traffic update...")
try:
    update_traffic()
    print("✅ Manual update completed successfully!")
except Exception as e:
    print(f"❌ Error during manual update: {e}")
    import traceback
    traceback.print_exc()
