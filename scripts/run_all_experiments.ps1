$ErrorActionPreference = "Stop"

function Invoke-PoetryChecked {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CommandArguments
    )
    & poetry @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Poetry command failed with exit code $LASTEXITCODE."
    }
}

$allowedArguments = @("--quick", "--full", "--no-render")
$unknownArguments = @($args | Where-Object { $_ -notin $allowedArguments })
if ($unknownArguments.Count -gt 0) {
    throw "Unknown arguments: $($unknownArguments -join ', ')"
}

$mode = if ($args -contains "--quick") { "--quick" } else { "--full" }
$renderNotebooks = $args -notcontains "--no-render"

Invoke-PoetryChecked run python scripts/build_experiment_data.py
$clippingModes = if ($mode -eq "--full") {
    @("--with-clipping", "--without-clipping")
} else {
    @("--without-clipping")
}

$evaluationStructures = if ($mode -eq "--full") {
    @("fixed", "rolling_5y")
} else {
    @("rolling_5y")
}

foreach ($evaluationStructure in $evaluationStructures) {
    foreach ($clippingMode in $clippingModes) {
        Invoke-PoetryChecked run python scripts/experiment_factors.py $mode --evaluation-structure $evaluationStructure $clippingMode
        Invoke-PoetryChecked run python scripts/experiment_predict_loss.py $mode --evaluation-structure $evaluationStructure $clippingMode
        Invoke-PoetryChecked run python scripts/experiment_tail_risk.py $mode --evaluation-structure $evaluationStructure $clippingMode
    }
}
Invoke-PoetryChecked run python scripts/experiment_distortion_risk.py
Invoke-PoetryChecked run python scripts/experiment_autoregression.py
Invoke-PoetryChecked run python scripts/audit_results.py

if ($renderNotebooks) {
    & "$PSScriptRoot\render_notebooks.ps1"
}
