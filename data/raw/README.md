# Raw-data provenance

Files in this directory are immutable inputs. Do not edit vendor files in place;
replace them with a new dated snapshot and rebuild `data/experiments/`.

- `bloomberg/` contains licensed Bloomberg exports. Confirm redistribution rights
  before publishing or sharing the repository. Prefer distributing checksums and
  acquisition instructions when the underlying files cannot be redistributed.
- `fred/` contains downloaded Federal Reserve Economic Data inputs. Preserve the
  series identifier and download date in each filename.

`scripts/build_experiment_data.py` hashes every raw input and records those hashes
in `data/experiments/manifest.json`. Experiment runs record the resulting data
fingerprint in their own manifests.
