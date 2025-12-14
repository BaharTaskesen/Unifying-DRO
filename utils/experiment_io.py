# utils/experiment_io.py
import json
import os
from datetime import datetime
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_name(tag: str) -> str:
    # timestamp makes runs unique even with same config
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}"

def save_config_json(cfg_dict: Dict[str, Any], out_dir: str, run_name: str) -> None:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{run_name}_config.json")
    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)

def savez_results(arr_dict: Dict[str, Any], out_dir: str, run_name: str) -> str:
    import numpy as np

    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{run_name}.npz")
    np.savez(path, **arr_dict)
    return path
