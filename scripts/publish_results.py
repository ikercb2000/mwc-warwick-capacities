"""Promote a successful immutable run into the tracked dissertation snapshot."""

from __future__ import annotations

# Publication remains an explicit command and never occurs during experimentation.

import argparse

from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import experiment_catalog, publish_run_snapshot


paths = RepoPaths.discover()
catalog = experiment_catalog(paths.root)
parser = argparse.ArgumentParser(
    description="Publish one successful experiment run to flat result tables/figures."
)
parser.add_argument("experiment", choices=sorted(catalog))
parser.add_argument(
    "--run-id",
    help="Run to publish; defaults to the latest successful run.",
)
args = parser.parse_args()
publication = publish_run_snapshot(
    paths,
    experiment_id=args.experiment,
    run_id=args.run_id,
)
print(
    f"Published {len(publication['artifacts'])} artifacts from "
    f"{publication['experiment_id']}/{publication['run_id']}."
)
