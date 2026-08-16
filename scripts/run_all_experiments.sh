#!/usr/bin/env bash

set -euo pipefail

mode="--full"
render_notebooks=true
mode_was_selected=false

for argument in "$@"; do
    case "$argument" in
        --quick|--full)
            if [[ "$mode_was_selected" == true && "$mode" != "$argument" ]]; then
                echo "--quick and --full are mutually exclusive." >&2
                exit 2
            fi
            mode="$argument"
            mode_was_selected=true
            ;;
        --no-render)
            render_notebooks=false
            ;;
        *)
            echo "Unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_dir/.." && pwd)"
cd "$repository_root"

poetry run python scripts/build_experiment_data.py
poetry run python scripts/experiment_factors.py "$mode"
poetry run python scripts/experiment_predict_loss.py "$mode"
poetry run python scripts/experiment_tail_risk.py "$mode"
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
poetry run python scripts/audit_results.py

if [[ "$render_notebooks" == true ]]; then
    bash "$script_dir/render_notebooks.sh"
fi
