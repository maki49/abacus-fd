# ABACUS Finite Difference (abacus-fd)

Python library for computing forces via finite difference with ABACUS DFT software.

## Package Structure

- `abacus_fd/` - Main package
  - `core.py` - Core functions
  - `__init__.py` - CLI interface and exports
- `abacus_finidiff.py` - Legacy single-file version (backward compatibility)

## Installation

```bash
pip install .
```

After installation, use the CLI:
```bash
abacus-fd --help
```

## Key Functions (CLI Commands)

- `gs-all` - Ground state forces for all atoms
- `lr-all` - LR-TDDFT excited state forces for all atoms
- `gs-custom` - Ground state forces for specified atoms
- `lr-custom` - LR-TDDFT excited state forces for specified atoms
- `kslr-custom` - KS-LR excited state forces for specified atoms
- `kslr-all` - KS-LR excited state forces for all atoms

## Python API

```python
from abacus_fd import (
    run_diff_all_groundstate,
    run_diff_all_lr,
    run_diff_custom_groundstate,
    run_diff_custom_lr,
    run_diff_custom_kslr,
    run_diff_all_kslr,
)
```

## Notes

- Central difference is default (`central=True` splits displacement into +dx/2 and -dx/2)
- `run_diff_*_lr` requires both `INPUT_gs` and `INPUT_lr` files
- Output suffix defaults to "ABACUS" if not specified in INPUT files
- Atom indices in CLI are 0-based
