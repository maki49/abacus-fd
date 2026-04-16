# ABACUS Finite Difference (abacus-fd)

A Python library for computing forces (unit: eV/Angstrom) via finite difference with ABACUS DFT software.

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
- `ase` with ABACUS IO support (auto-installed from `https://gitlab.com/1041176461/ase-abacus.git` by `pip install .`)
- `numpy`
- ABACUS executable (external, not included)

## Command Line Interface

```bash
abacus-fd --help
```

### Command Categories

| Command | Description | Atoms | Directions |
|---------|-------------|-------|------------|
| `kslr-all` | KSDFT + LR-TDDFT excited state forces | All atoms | x, y, z |
| `kslr-custom` | KSDFT + LR-TDDFT excited state forces | Specified | Specified |
| `gs-all` | Ground state forces | All atoms | x, y, z |
| `gs-custom` | Ground state forces | Specified | Specified |
| `lr-all` | LR-TDDFT excited state forces | All atoms | x, y, z |
| `lr-custom` | LR-TDDFT excited state forces | Specified | Specified |

- `all`: Compute forces for all atoms, displacement directions are x, y, z
- `custom`: Compute forces only for specified atoms and directions
- Atom indices are 0-based
- Central difference (displacement ±dx/2)

### Common Parameters

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-d` | `--dir` | Working directory | Current dir |
| `-a` | `--abacus` | ABACUS executable path | `abacus` |
| `-x` | `--dx` | Displacement (Angstrom) | `0.001` |
| `-i` | `--indices` | Atom indices (comma-separated) | - |
| `-s` | `--skip-gs` | Skip ground state calculation | False |

### kslr-all

KSDFT + LR-TDDFT excited state forces for all atoms along x/y/z.

**Usage:**
```bash
abacus-fd kslr-all [-d DIR] [-a ABACUS] [-x DX]
```

**Examples:**
```bash
abacus-fd kslr-all
abacus-fd kslr-all -d /path/to/dir -a /path/to/abacus -x 0.001
```

**Input files:** `STRU`, `INPUT` (must set `lr_nstates`, `esolver_type ks-lr`)

**Output files:**
- `excited_forces.npy`: numpy array, shape `(2, nstates, natoms, 3)`
- `excited_forces.txt`: text format, columns `S/T  state_idx  atom_idx  x  y  z`

---

### kslr-custom

KSDFT + LR-TDDFT excited state forces for specified atoms and directions.

**Usage:**
```bash
abacus-fd kslr-custom [-d DIR] [-a ABACUS] -i INDICES --axes AXES [-x DX]
```

**Python equivalent:**
```python
from abacus_fd import run_diff_custom_kslr
run_diff_custom_kslr(dir=".", abacus_path="abacus", diffed_atom_indices=[0, 1], axes=['x', 'y'], dx=0.001)
```

**Examples:**
```bash
abacus-fd kslr-custom -i 0,1 --axes x,y
abacus-fd kslr-custom -d /path -a /abacus -i 5 --axes z
```

**Input files:** `STRU`, `INPUT` (must set `lr_nstates`, `esolver_type ks-lr`)

---

### gs-all

Ground state forces for all atoms along x/y/z.

**Usage:**
```bash
abacus-fd gs-all [-d DIR] [-a ABACUS] [-x DX]
```

**Examples:**
```bash
abacus-fd gs-all
abacus-fd gs-all -d /path/to/dir -x 0.001
```

**Input files:** `STRU`, `INPUT`

---

### gs-custom

Ground state forces for specified atoms and directions.

**Usage:**
```bash
abacus-fd gs-custom [-d DIR] [-a ABACUS] -i INDICES --axes AXES [-x DX]
```

**Python equivalent:**
```python
from abacus_fd import run_diff_custom_groundstate
run_diff_custom_groundstate(dir=".", abacus_path="abacus", diffed_atom_indices=[0, 1, 2], axes=['x', 'y'], dx=0.001)
```

**Examples:**
```bash
abacus-fd gs-custom -i 0,1,2 --axes x,y
abacus-fd gs-custom -d /path -i 5 --axes z
```

**Input files:** `STRU`, `INPUT`

---

### lr-all

LR-TDDFT excited state forces for all atoms along x/y/z.

**Usage:**
```bash
abacus-fd lr-all [-d DIR] [-a ABACUS] [-x DX] [-s]
```

**Examples:**
```bash
abacus-fd lr-all
abacus-fd lr-all -d /path -x 0.001
abacus-fd lr-all -d /path --skip-gs
```

**Input files:** `STRU`, `INPUT_gs`, `INPUT_lr`(must set `lr_nstates`, `esolver_type lr`)

**Output files:**
- `excited_forces.npy`: numpy array, shape `(2, nstates, natoms, 3)`
- `excited_forces.txt`: text format, columns `S/T  state_idx  atom_idx  x  y  z`

---

### lr-custom

LR-TDDFT excited state forces for specified atoms and directions.

**Usage:**
```bash
abacus-fd lr-custom [-d DIR] [-a ABACUS] -i INDICES --axes AXES [-x DX] [-s]
```

**Python equivalent:**
```python
from abacus_fd import run_diff_custom_lr
run_diff_custom_lr(dir=".", abacus_path="abacus", diffed_atom_indices=[1], axes=['z'], dx=0.001, skip_groundstate=False)
```

**Examples:**
```bash
abacus-fd lr-custom -i 0,1 --axes x,y,z
abacus-fd lr-custom -d /path -i 5 --axes z --skip-gs
```

**Input files:** `STRU`, `INPUT_gs`, `INPUT_lr`(must set `lr_nstates`, `esolver_type lr`)

---

## Notes

- Central difference is used by default (splits displacement into +dx/2 and -dx/2)
- `lr-*` commands require both `INPUT_gs` and `INPUT_lr` files
- Output suffix defaults to "ABACUS" if not specified in INPUT files
- Atom indices are 0-based (first atom is index 0)

## Testing

No formal test suite. Test data in `test/` directory is for manual verification with real ABACUS runs.
