"""Audit every notebook reference against latest successful experiment runs."""

from __future__ import annotations

# This audit intentionally remains a directly executable script.

from mwc_experiments.reporting import audit_notebook_artifacts


audit = audit_notebook_artifacts()
print(audit.to_string(index=False))
missing = audit.loc[~audit["exists"]]
if not missing.empty:
    raise SystemExit(f"{len(missing)} notebook artifacts are missing.")
print(f"All {len(audit)} notebook artifacts are available.")
