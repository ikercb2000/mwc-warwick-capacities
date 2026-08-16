$ErrorActionPreference = "Stop"

$mode = if ($args -contains "--quick") { "--quick" } else { "--full" }

poetry run python scripts/build_experiment_data.py
poetry run python scripts/experiment_factors.py $mode
poetry run python scripts/experiment_predict_loss.py $mode
poetry run python scripts/experiment_tail_risk.py $mode
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
poetry run python scripts/audit_results.py
