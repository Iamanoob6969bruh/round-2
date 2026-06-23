import os
import sys
from pathlib import Path
import modal

# Insert backend root to sys.path so backend imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modal_app import image

app = modal.App("test-db")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("trafficviolens-secrets")]
)
def run_test():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import json
    import traceback
    
    db_url = os.environ.get("DATABASE_URL")
    print("DB_URL is set:", bool(db_url))
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("Testing casting and subtraction operator...")
        query = "SELECT CAST(record_json AS jsonb) - 'evidence_image' - 'plate_crop_image' AS record_json FROM violations ORDER BY id ASC LIMIT 1"
        cur.execute(query)
        row = cur.fetchone()
        print("Success! Row keys:", row.keys() if row else "None")
        if row:
            # We want to check the type of record_json returned
            val = row["record_json"]
            print("Type of record_json returned:", type(val))
            print("Preview of record_json:", str(val)[:200])
            if isinstance(val, str):
                data = json.loads(val)
            else:
                data = val
            print("evidence_image in record:", "evidence_image" in data)
            print("plate_crop_image in record:", "plate_crop_image" in data)
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    # run locally
    pass
