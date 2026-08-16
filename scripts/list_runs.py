"""List local experiment runs and identify the currently published latest run."""

from __future__ import annotations

# This inspection intentionally remains a directly executable script.

import json

import pandas as pd

from mwc_experiments.paths import RepoPaths


paths = RepoPaths.discover()
latest = {
    pointer.stem: json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    for pointer in paths.latest.glob("*.json")
}
rows: list[dict[str, object]] = []
for manifest_path in sorted(paths.runs.glob("*/*/manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest["experiment_id"]
    rows.append(
        {
            "experiment": experiment,
            "run_id": manifest["run_id"],
            "mode": manifest["mode"],
            "status": manifest["status"],
            "latest": latest.get(experiment) == manifest["run_id"],
            "started UTC": manifest["started_at_utc"],
            "duration seconds": manifest.get("duration_seconds"),
            "git commit": manifest.get("git", {}).get("commit"),
            "dirty": manifest.get("git", {}).get("dirty"),
        }
    )
catalogue = pd.DataFrame(rows)
print(catalogue.to_string(index=False) if not catalogue.empty else "No runs found.")
