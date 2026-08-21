$ErrorActionPreference = "Stop"

$allowedArguments = @("--quick", "--full", "--no-render")
$unknownArguments = @($args | Where-Object { $_ -notin $allowedArguments })
if ($unknownArguments.Count -gt 0) {
    throw "Unknown arguments: $($unknownArguments -join ', ')"
}

$mode = if ($args -contains "--quick") { "--quick" } else { "--full" }
$renderNotebooks = $args -notcontains "--no-render"

poetry run python scripts/build_experiment_data.py
$clippingModes = if ($mode -eq "--full") {
    @("--with-clipping", "--without-clipping")
} else {
    @("--without-clipping")
}

foreach ($clippingMode in $clippingModes) {
    poetry run python scripts/experiment_factors.py $mode $clippingMode
    poetry run python scripts/experiment_predict_loss.py $mode $clippingMode
    poetry run python scripts/experiment_tail_risk.py $mode $clippingMode
}
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
poetry run python scripts/audit_results.py

if ($renderNotebooks) {
    & "$PSScriptRoot\render_notebooks.ps1"
}
