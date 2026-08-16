$ErrorActionPreference = "Stop"

$allowedArguments = @("--quick", "--full", "--no-render")
$unknownArguments = @($args | Where-Object { $_ -notin $allowedArguments })
if ($unknownArguments.Count -gt 0) {
    throw "Unknown arguments: $($unknownArguments -join ', ')"
}

$mode = if ($args -contains "--quick") { "--quick" } else { "--full" }
$renderNotebooks = $args -notcontains "--no-render"

poetry run python scripts/build_experiment_data.py
poetry run python scripts/experiment_factors.py $mode
poetry run python scripts/experiment_predict_loss.py $mode
poetry run python scripts/experiment_tail_risk.py $mode
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
poetry run python scripts/audit_results.py

if ($renderNotebooks) {
    & "$PSScriptRoot\render_notebooks.ps1"
}
