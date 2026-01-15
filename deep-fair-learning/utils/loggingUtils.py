import json
import os

def save_metrics_json(outpath: str, metrics: dict):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(metrics, f, indent=4)
