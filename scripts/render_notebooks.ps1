$ErrorActionPreference = "Stop"

$repositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$notebookDirectory = Join-Path $repositoryRoot "notebooks"
$outputDirectory = Join-Path $repositoryRoot "data\results\reports"

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$notebooks = Get-ChildItem -Path $notebookDirectory -Filter "*.ipynb" | Sort-Object Name

if ($notebooks.Count -eq 0) {
    throw "No notebooks were found in $notebookDirectory."
}

Push-Location $repositoryRoot
try {
    foreach ($notebook in $notebooks) {
        Write-Host "Rendering $($notebook.Name)..."
        poetry run jupyter nbconvert `
            --to html `
            --execute $notebook.FullName `
            --output-dir $outputDirectory `
            --ExecutePreprocessor.timeout=600 `
            --log-level=WARN
        if ($LASTEXITCODE -ne 0) {
            throw "Notebook rendering failed: $($notebook.Name)"
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Rendered $($notebooks.Count) notebooks to $outputDirectory"
