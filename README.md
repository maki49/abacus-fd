# ABACUS Finite Difference (abacus-fd)

A Python library for computing forces via finite difference with ABACUS DFT software.

## Installation

```bash
pip install .
```

Or install in development mode:

```bash
pip install -e .
```

## Dependencies

- `ase` (Atomic Simulation Environment) for reading/writing ABACUS STRU files
- `numpy`
- ABACUS executable (external, not included)

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

### run_diff_all_groundstate(dir, abacus_path, dx=0.001)

Compute ground state forces for all atoms using finite difference.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `dx`: Displacement distance in Angstrom (default: 0.001)

**Returns:** numpy array of shape (natoms, 3) containing forces in eV/Angstrom

### run_diff_all_lr(dir, abacus_path, dx=0.001, skip_groundstate=False)

Compute LR-TDDFT excited state forces for all atoms.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `dx`: Displacement distance in Angstrom (default: 0.001)
- `skip_groundstate`: Skip ground state calculation if already done (default: False)

**Returns:** numpy array of shape (2, nstates, natoms, 3)
- `[0]` = singlet forces, `[1]` = triplet forces, in eV/Angstrom

**Output files** (saved to `dir`):
- `excited_forces.npy`: numpy array
- `excited_forces.txt`: human-readable format
  - Columns: `S/T(0=S,1=T)  state_idx  atom_idx  x  y  z`

### run_diff_custom_groundstate(dir, abacus_path, diffed_atom_indices, axes, dx=0.001)

Compute ground state forces for specified atoms using finite difference.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `diffed_atom_indices`: List of atom indices (0-based) to displace
- `axes`: List of axes to displace ('x', 'y', 'z')
- `dx`: Displacement distance in Angstrom (default: 0.001)

**Returns:** dict mapping atom index to axis to force array

### run_diff_custom_lr(dir, abacus_path, diffed_atom_indices, axes, dx=0.001, skip_groundstate=False)

Compute LR-TDDFT excited state forces for specified atoms.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `diffed_atom_indices`: List of atom indices (0-based) to displace
- `axes`: List of axes to displace ('x', 'y', 'z')
- `dx`: Displacement distance in Angstrom (default: 0.001)
- `skip_groundstate`: Skip ground state calculation if already done (default: False)

**Returns:** dict mapping atom index to axis to force array

### run_diff_custom_kslr(dir, abacus_path, diffed_atom_indices, axes, dx=0.001)

Compute excited state forces using Kohn-Sham LR for specified atoms.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `diffed_atom_indices`: List of atom indices (0-based) to displace
- `axes`: List of axes to displace ('x', 'y', 'z')
- `dx`: Displacement distance in Angstrom (default: 0.001)

**Returns:** dict mapping atom index to axis to force array

### run_diff_all_kslr(dir, abacus_path, dx=0.001)

Compute excited state forces using Kohn-Sham LR for all atoms.

**Parameters:**
- `dir`: Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files
- `abacus_path`: Path to the ABACUS executable
- `dx`: Displacement distance in Angstrom (default: 0.001)

**Returns:** numpy array of shape (2, nstates, natoms, 3)
- `[0]` = singlet forces, `[1]` = triplet forces, in eV/Angstrom

**Output files** (saved to `dir`):
- `excited_forces.npy`: numpy array
- `excited_forces.txt`: human-readable format
  - Columns: `S/T(0=S,1=T)  state_idx  atom_idx  x  y  z`

---

## Command Line Interface

After installation, the `abacus-fd` command is available:

```bash
abacus-fd --help
```

### Commands

#### gs-all

Compute ground state forces for all atoms using finite difference.

```bash
abacus-fd gs-all DIR ABACUS [--dx DX]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)

**Example:**
```bash
abacus-fd gs-all /path/to/dir /path/to/abacus --dx 0.001
```

---

#### lr-all

Compute LR-TDDFT excited state forces for all atoms.

```bash
abacus-fd lr-all DIR ABACUS [--dx DX] [--skip-gs]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)
- `--skip-gs`: Skip ground state calculation if already done

**Example:**
```bash
abacus-fd lr-all /path/to/dir /path/to/abacus --dx 0.001
abacus-fd lr-all /path/to/dir /path/to/abacus --skip-gs
```

---

#### gs-custom

Compute ground state forces for specified atoms using finite difference.

```bash
abacus-fd gs-custom DIR ABACUS --indices INDICES --axes AXES [--dx DX]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--indices INDICES`: Comma-separated list of atom indices (0-based)
- `--axes AXES`: Comma-separated list of axes (x,y,z)
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)

**Example:**
```bash
abacus-fd gs-custom /path/to/dir /path/to/abacus --indices 0,1,2 --axes x,y
abacus-fd gs-custom /path/to/dir /path/to/abacus --indices 5 --axes z
```

---

#### lr-custom

Compute LR-TDDFT excited state forces for specified atoms.

```bash
abacus-fd lr-custom DIR ABACUS --indices INDICES --axes AXES [--dx DX] [--skip-gs]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--indices INDICES`: Comma-separated list of atom indices (0-based)
- `--axes AXES`: Comma-separated list of axes (x,y,z)
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)
- `--skip-gs`: Skip ground state calculation if already done

**Example:**
```bash
abacus-fd lr-custom /path/to/dir /path/to/abacus --indices 0,1 --axes x,y,z
abacus-fd lr-custom /path/to/dir /path/to/abacus --indices 5 --axes z --skip-gs
```

---

#### kslr-custom

Compute excited state forces using Kohn-Sham LR for specified atoms.

```bash
abacus-fd kslr-custom DIR ABACUS --indices INDICES --axes AXES [--dx DX]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--indices INDICES`: Comma-separated list of atom indices (0-based)
- `--axes AXES`: Comma-separated list of axes (x,y,z)
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)

**Example:**
```bash
abacus-fd kslr-custom /path/to/dir /path/to/abacus --indices 0,1 --axes x,y
```

---

#### kslr-all

Compute excited state forces using Kohn-Sham LR for all atoms.

```bash
abacus-fd kslr-all DIR ABACUS [--dx DX]
```

**Arguments:**
- `DIR`: Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files
- `ABACUS`: Path to the ABACUS executable
- `--dx DX`: Displacement distance in Angstrom (default: 0.001)

**Example:**
```bash
abacus-fd kslr-all /path/to/dir /path/to/abacus
```

---

## Notes

- Central difference is used by default (splits displacement into +dx/2 and -dx/2)
- `lr-*` commands require both `INPUT_gs` and `INPUT_lr` files
- Output suffix defaults to "ABACUS" if not specified in INPUT files
- Atom indices are 0-based (first atom is index 0)

## Testing

No formal test suite. Test data in `test/` directory is for manual verification with real ABACUS runs.
