import csv
import os
import threading
from datetime import datetime, timezone

CSV_PATH = os.path.join("database", "toxic_database.csv")

# Keep this in sync with LABELS in main.py
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

FIELDNAMES = (
    ["timestamp", "description"]
    + [f"bert_{label}" for label in LABELS]
    + [f"clip_{label}" for label in LABELS]
    + [f"fused_{label}" for label in LABELS]
    + ["status"]
)

# Concurrent requests run this in different threads (via asyncio.to_thread),
# so guard the append with a lock to avoid two writes interleaving mid-row.
_lock = threading.Lock()


def log_prediction(description, bert_scores, clip_scores, fused_scores, status="success"):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "status": status,
    }
    for label in LABELS:
        row[f"bert_{label}"] = bert_scores.get(label) if bert_scores else None
        row[f"clip_{label}"] = clip_scores.get(label) if clip_scores else None
        row[f"fused_{label}"] = fused_scores.get(label) if fused_scores else None

    with _lock:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        file_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def read_all_predictions():
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))