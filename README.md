# S04-AES-Hawkes reproducibility materials

This package contains reproducibility materials for the associated manuscript.

This deliberately narrow package contains the standard-library Python certificate sources, configurations, tests, and frozen JSON outputs needed to reproduce the mathematical certificate calculations behind the author-approved v0.6 manuscript. It excludes the private S1 ZIP, internal Controller/audit records, prompts, author contact data, and local repository paths.

## Layout

- `code/workpackages/WP3_count/` and `code/workpackages/WP4_timestamps/`: protected exact-rational Hawkes certificate runners plus original configurations and auxiliary formula modules.
- `code/workpackages/S04_P1_A_MECHANISM_ATTRIBUTION/`: exact-rational dependence-increment certificate and focused unit checks.
- `code/physical_poisson/`: the separate diagnostic Poisson score-ablation certificate and focused checks.
- `results/`: byte-preserved frozen JSON outputs from the authenticated private evidence.

## Requirements and reproduction

Python 3.13.x with only the standard library was the recorded certificate environment. No network access, package installation, random seed, or private repository is required.

From `code/`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
Push-Location workpackages\S04_P1_A_MECHANISM_ATTRIBUTION
python -B -m unittest -v test_mechanism_feasibility.py
python -B mechanism_feasibility_certificate.py
Pop-Location
Push-Location physical_poisson
python -B -m unittest -v test_ablation_certificate.py
python -B ablation_certificate.py
Pop-Location
```

The Hawkes runners rewrite their local `raw_results.json` when executed. Run them only in a scratch copy, not in a checksum-verified archival copy:

```powershell
python -B workpackages\WP3_count\certificate_counts.py
python -B workpackages\WP4_timestamps\certificate_timestamps.py
```

## Scope and limitations

The package supports the protected, finite-horizon certificate at the manuscript's stated parameter triple. It does not establish a general Hawkes mechanism law, theta monotonicity, excitation necessity, Blackwell dominance, a wider parameter/kernel result, exact optimized gaps, or an empirical effect size.

