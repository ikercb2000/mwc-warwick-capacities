#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_dir/.." && pwd)"
notebook_directory="$repository_root/notebooks"
output_directory="$repository_root/data/results/reports"

mkdir -p "$output_directory"
shopt -s nullglob
notebooks=("$notebook_directory"/*.ipynb)

if (( ${#notebooks[@]} == 0 )); then
    echo "No notebooks were found in $notebook_directory." >&2
    exit 1
fi

cd "$repository_root"
for notebook in "${notebooks[@]}"; do
    echo "Rendering $(basename "$notebook")..."
    poetry run jupyter nbconvert \
        --to html \
        --execute "$notebook" \
        --output-dir "$output_directory" \
        --ExecutePreprocessor.timeout=600 \
        --log-level=WARN
done

echo "Rendered ${#notebooks[@]} notebooks to $output_directory"
