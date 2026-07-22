import csv
import os
import threading
from datetime import datetime, timezone

CSV_PATH = os.path.join("database", "toxic_database.csv")

# Keep this in sync with LABELS in main.py
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

FIELDNAMES = ["timestamp", "source"] + LABELS

# Concurrent requests run this in different threads (via asyncio.to_thread),
# so guard the append with a lock to avoid two writes interleaving mid-row.
_lock = threading.Lock()


def log_prediction(source: str, flags: dict):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    for label in LABELS:
        row[label] = bool(flags.get(label, False))

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