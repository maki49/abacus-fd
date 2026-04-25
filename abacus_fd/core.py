import concurrent.futures
import subprocess
from ase.io import read, write
from ase.io.formats import UnknownFileTypeError
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def _read_abacus(path):
    """Read ABACUS STRU file via ASE with a clear dependency error."""
    try:
        return read(path, format="abacus", verbose=True)
    except UnknownFileTypeError as exc:
        raise RuntimeError(
            "ASE format 'abacus' is unavailable. Please install ASE with ABACUS "
            "format support in the current Python environment."
        ) from exc


def _write_abacus(path, atoms, scaled=False):
    """Write ABACUS STRU file via ASE with a clear dependency error."""
    try:
        basis = atoms.info.get("basis")
        pp = atoms.info.get("pp")
        write(path, atoms, format="abacus", scaled=scaled, basis=basis, pp=pp)
    except UnknownFileTypeError as exc:
        raise RuntimeError(
            "ASE format 'abacus' is unavailable."
        ) from exc


def grep_parameter_from_input(input_file, para_name):
    """Pure Python implementation to grep parameter from abacus INPUT file."""
    if not os.path.exists(input_file):
        return None
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].lower() == para_name.lower() and len(parts) >= 2:
                return parts[1]
    return None


def modify_input_calculation(input_file, target_calc="scf"):
    """Set calculation type in the given INPUT file while preserving user solver settings."""
    if not os.path.exists(input_file):
        return
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    # Only force the calculation type. Preserve others like ks_solver, lr_solver.
    forced_params = {
        "calculation": target_calc,
    }
    handled_params = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
            
        parts = stripped.split()
        param_name = parts[0].lower()
        
        if param_name in forced_params:
            new_line = f"{param_name.ljust(15)} {forced_params[param_name]}\n"
            new_lines.append(new_line)
            handled_params.add(param_name)
        else:
            new_lines.append(line)
    
    for param_name, value in forced_params.items():
        if param_name not in handled_params:
            new_lines.insert(1, f"{param_name.ljust(15)} {value}\n")
        
    with open(input_file, 'h' if False else 'w') as f: # Use 'w'
        f.writelines(new_lines)


def grep_groundstate_energy_from_log(log):
    """Pure Python implementation to grep groundstate energy (eV) from abacus log file."""
    if not os.path.exists(log):
        raise FileNotFoundError(f"Log file not found: {log}")
        
    e = None
    last_ks_energy = None
    
    with open(log, "r") as f:
        for line in f:
            if "!FINAL_ETOT_IS" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try: e = float(parts[1])
                    except ValueError: pass
            elif "E_KohnSham" in line:
                parts = line.split()
                if len(parts) >= 3:
                    try: last_ks_energy = float(parts[2])
                    except ValueError: pass
    
    if e is not None: return e
    if last_ks_energy is not None: return last_ks_energy
    raise ValueError(f"Cannot parse energy from log file {log}.")


def grep_excitation_energy_from_log(log, nstate, nspin=None):
    """Pure Python implementation to grep LR-TDDFT excitation energy (eV)."""
    if not os.path.exists(log):
        raise FileNotFoundError(f"Log file not found: {log}")
        
    energies = []
    with open(log, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "Excitation Energy" in line:
            for j in range(1, nstate + 2):
                if i + j < len(lines):
                    parts = lines[i+j].split()
                    if len(parts) >= 3:
                        try: energies.append(float(parts[2]))
                        except ValueError: continue
            if energies: break
            
    if not energies:
        for i, line in enumerate(lines):
            if "eigenvalues: (eV)" in line:
                if i + 1 < len(lines):
                    parts = lines[i+1].split()
                    for p in parts:
                        try: energies.append(float(p))
                        except ValueError: continue
                if energies: break

    if not energies:
        raise ValueError(f"No excitation energies found in {log}")
        
    e_arr = np.array(energies)
    if len(e_arr) == nstate:
        return np.array([e_arr, np.zeros_like(e_arr)]).tolist()
    
    try: return e_arr.reshape(2, -1).tolist()
    except ValueError:
        if len(e_arr) >= nstate:
            s = e_arr[:nstate]
            t = e_arr[nstate:2*nstate] if len(e_arr) >= 2*nstate else np.zeros_like(s)
            return np.array([s, t]).tolist()
        raise


def grep_forces_from_log(log, natom):
    """Pure Python implementation to grep analytical forces from abacus log file."""
    if not os.path.exists(log):
        return None
    
    forces = np.zeros((natom, 3))
    found = False
    with open(log, "r") as f:
        lines = f.readlines()
        
    for i in range(len(lines) - 1, -1, -1):
        if "TOTAL-FORCE (eV/Angstrom)" in lines[i]:
            # ABACUS LCAO Force Table format:
            # i  : #TOTAL-FORCE (eV/Angstrom)#
            # i+1: -------------------------------------------------------------------------
            # i+2:     Atoms              Force_x              Force_y              Force_z 
            # i+3: -------------------------------------------------------------------------
            # i+4:       Li1         1.2496801433         0.0000000000         0.0000000000 
            start_data = i + 4
            for j in range(natom):
                if start_data + j < len(lines):
                    parts = lines[start_data + j].split()
                    if len(parts) >= 4:
                        try:
                            # parts[0] is AtomLabel, parts[1:4] are fx, fy, fz
                            forces[j] = [float(parts[1]), float(parts[2]), float(parts[3])]
                        except ValueError: continue
            found = True
            break
        elif "!FORCE_IS" in lines[i]:
            # Handle PW or other formats if needed
            found = True
            break
            
    return forces if found else None


def move_an_atom_in_stru(src, dest, atom_index, dr, scaled=False):
    """Move an atom in the `src` STRU file and write to `dest`."""
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    atoms = _read_abacus(src)
    positions = atoms.get_positions()
    positions[atom_index] += np.array(dr)
    atoms.set_positions(positions)
    
    orb = ""
    with open(src, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "NUMERICAL_ORBITAL" in line:
                if i + 1 < len(lines): orb = lines[i+1].strip()
                break
                
    _write_abacus(dest, atoms, scaled=scaled)
    if orb:
        with open(dest, "r") as f: d_lines = f.readlines()
        for i, line in enumerate(d_lines):
            if line.strip() == "NUMERICAL_ORBITAL":
                d_lines[i + 1] = f"{orb}\n"
                break
        with open(dest, "w") as f: f.writelines(d_lines)


def run_abacus(dir, abacus, nproc=1, log="log.txt"):
    """Run ABACUS using subprocess for better reliability and environment cleanup."""
    absdir = os.path.abspath(dir)
    executable = os.path.abspath(abacus)
    
    # Aggressively clean environment to prevent MPI deadlocks
    env = os.environ.copy()
    mpi_prefixes = ["OMPI_", "PMIX_", "PMI_", "HYDRA_", "MPI_", "I_MPI_", "MV2_", "UCX_", "OPAL_"]
    for k in list(env.keys()):
        if any(k.startswith(prefix) for prefix in mpi_prefixes) or k == "LD_PRELOAD":
            env.pop(k, None)
            
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MKL_SERIAL"] = "YES"
    
    if nproc > 1:
        cmd = ["mpirun", "-np", str(nproc), executable]
    else:
        cmd = [executable]
        
    try:
        with open(os.path.join(absdir, log), "w") as out:
            subprocess.run(cmd, cwd=absdir, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ABACUS failed in {absdir} with return code {e.returncode}")
        raise RuntimeError(f"ABACUS failed with status {e.returncode}.")


def run_single_kslr(dir=".", abacus_path="abacus", nproc=1):
    """Run a single SCF + LR-TDDFT calculation to get states and forces."""
    dir = os.path.abspath(dir)
    src_input = os.path.join(dir, "INPUT")
    src_stru = os.path.join(dir, "STRU")
    if not os.path.exists(src_input):
        raise FileNotFoundError(f"INPUT file not found in {dir}")

    # Get natom from STRU for force parsing later
    atoms = _read_abacus(src_stru)
    natom = len(atoms)

    # Ensure output parameters are set for FSSH read-back
    modify_input_calculation(src_input, "scf")
    with open(src_input, "r") as f:
        lines = f.readlines()
    
    # Ensure cal_force 1 is set to get analytical ground state forces
    extra_params = {
        "out_wfc_lcao": "1",
        "out_wfc_lr": "True",
        "cal_force": "1",
    }
    
    new_lines = []
    handled = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        parts = stripped.split()
        param = parts[0].lower()
        if param in extra_params:
            new_lines.append(f"{param.ljust(15)} {extra_params[param]}\n")
            handled.add(param)
        else:
            new_lines.append(line)
    for p, v in extra_params.items():
        if p not in handled:
            new_lines.insert(1, f"{p.ljust(15)} {v}\n")
            
    with open(src_input, "w") as f:
        f.writelines(new_lines)

    logger.info(f"Running single ABACUS point in {dir} with nproc={nproc}...")
    run_abacus(dir, abacus_path, nproc=nproc, log="ks-lr.log")
    
    # [Hijack-Force] Extract and save forces from multiple log locations
    suffix = grep_parameter_from_input(src_input, "suffix") or "ABACUS"
    possible_logs = [
        os.path.join(dir, f"OUT.{suffix}", "running_scf.log"),
        os.path.join(dir, "ks-lr.log")
    ]
    
    forces = None
    for log_file in possible_logs:
        if os.path.exists(log_file):
            try:
                forces = grep_forces_from_log(log_file, natom)
                if forces is not None:
                    txt_file = os.path.join(dir, "ground_forces.txt")
                    with open(txt_file, "w") as f:
                        f.write(f"# Ground state forces extracted from {os.path.basename(log_file)}\n")
                        for i in range(natom):
                            f.write(f"{i}  {forces[i,0]:.12f}  {forces[i,1]:.12f}  {forces[i,2]:.12f}\n")
                    logger.info(f"Ground state forces saved to {txt_file}")
                    break
            except Exception as e:
                logger.debug(f"Failed to parse forces from {log_file}: {e}")
                
    if forces is None:
        logger.warning("Ground state forces could not be extracted from logs.")

    return True


def prepare_diff_all(src_stru, dx, output_dir="moved_STRU", central=True):
    """Prepare finite difference STRU files."""
    os.makedirs(output_dir, exist_ok=True)
    src_stru = os.path.abspath(src_stru)
    output_dir = os.path.abspath(output_dir)
    atoms = _read_abacus(src_stru)
    natoms = len(atoms)
    if central:
        direction_vecs = {
            "x": np.array([dx / 2, 0.0, 0.0]),
            "y": np.array([0.0, dx / 2, 0.0]),
            "z": np.array([0.0, 0.0, dx / 2]),
            "-x": np.array([-dx / 2, 0.0, 0.0]),
            "-y": np.array([0.0, -dx / 2, 0.0]),
            "-z": np.array([0.0, 0.0, -dx / 2]),
        }
    else:
        direction_vecs = {
            "x": np.array([dx, 0.0, 0.0]),
            "y": np.array([0.0, dx, 0.0]),
            "z": np.array([0.0, 0.0, dx]),
        }
    for i in range(natoms):
        for axis, dr in direction_vecs.items():
            dest = os.path.join(output_dir, f"STRU_{i}_{axis}")
            move_an_atom_in_stru(src_stru, dest, i, dr)


def _run_task_kslr_worker(task_info):
    """Worker function for a single ABACUS finite difference point."""
    task_dir = task_info['task_dir']
    abacus_path = task_info['abacus_path']
    nproc = task_info['nproc']
    log = task_info['log']
    src_input = task_info['src_input']
    task_stru_path = task_info['task_stru_path']
    src_kpt = task_info['src_kpt']
    task_name = task_info['task_name']
    calculation = task_info.get('calculation', 'scf')
    
    os.makedirs(task_dir, exist_ok=True)
    target_input = os.path.join(task_dir, 'INPUT')
    with open(src_input, "r") as fs, open(target_input, "w") as fd: 
        fd.write(fs.read())
    modify_input_calculation(target_input, calculation)
    
    with open(task_stru_path, "r") as fs, open(os.path.join(task_dir, "STRU"), "w") as fd:
        fd.write(fs.read())
    if src_kpt:
        with open(src_kpt, "r") as fs, open(os.path.join(task_dir, "KPT"), "w") as fd:
            fd.write(fs.read())
    
    run_abacus(task_dir, abacus_path, nproc=nproc, log=log)
    return task_name


def run_diff_all_kslr(dir=".", abacus_path="abacus", dx=0.001, nproc=1, nparallel=1, calculation="scf"):
    """Compute forces using task-level parallelism."""
    dir = os.path.abspath(dir)
    src_stru = os.path.join(dir, "STRU")
    src_input = os.path.join(dir, "INPUT")
    src_kpt = os.path.join(dir, "KPT") if os.path.exists(os.path.join(dir, "KPT")) else None

    # Prepare all STRU files
    prepare_diff_all(src_stru=src_stru, dx=dx, output_dir=os.path.join(dir, "moved_STRU"), central=True)
    
    atoms = _read_abacus(src_stru)
    natoms = len(atoms)
    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    tasks = []
    for i in range(natoms):
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            task_stru_path = os.path.join(dir, "moved_STRU", task_name)
            tasks.append({
                'task_name': task_name,
                'task_dir': task_dir,
                'abacus_path': abacus_path,
                'nproc': nproc,
                'log': "ks-lr.log",
                'src_input': src_input,
                'task_stru_path': task_stru_path,
                'src_kpt': src_kpt,
                'calculation': calculation
            })

    logger.info(f"Submitting {len(tasks)} tasks with nparallel={nparallel} (nproc per task={nproc})...")
    logger.warning("Resource Reminder: Ensure Total Cores >= nparallel * nproc to avoid oversubscription.")
    
    if nparallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nparallel) as executor:
            futures = [executor.submit(_run_task_kslr_worker, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    name = future.result()
                    logger.info(f"Finished task: {name}")
                except Exception as exc:
                    logger.error(f"Task generated an exception: {exc}")
                    raise
    else:
        for t in tasks:
            logger.info(f"Running ABACUS for {t['task_name']}...")
            _run_task_kslr_worker(t)

    # Collect results
    suffix = grep_parameter_from_input(src_input, "suffix") or "ABACUS"
    lr_nstates_str = grep_parameter_from_input(src_input, "lr_nstates")
    nstates = int(lr_nstates_str) if lr_nstates_str else 1

    ground_state_forces = np.zeros((natoms, 3))
    for i in range(natoms):
        energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_scf_log = os.path.join(points_dir, f"STRU_{i}_{axis}", f"OUT.{suffix}", "running_scf.log")
            energies.append(grep_groundstate_energy_from_log(task_scf_log))
        ground_state_forces[i] = (np.array(energies[3:]) - np.array(energies[:3])) / dx

    if calculation == "scf" and grep_parameter_from_input(src_input, "esolver_type") != "ks-lr":
        np.save(os.path.join(dir, "ground_forces.npy"), ground_state_forces)
        with open(os.path.join(dir, "ground_forces.txt"), "w") as f:
            for i in range(natoms):
                f.write(f"{i}  {ground_state_forces[i,0]:.6f}  {ground_state_forces[i,1]:.6f}  {ground_state_forces[i,2]:.6f}\n")
        return ground_state_forces

    excited_state_forces = np.zeros((natoms, 3, 2, nstates))
    for i in range(natoms):
        excitation_energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_scf_log = os.path.join(points_dir, f"STRU_{i}_{axis}", f"OUT.{suffix}", "running_scf.log")
            excitation_energies.append(grep_excitation_energy_from_log(task_scf_log, nstates))
        
        lr_force = (np.array(excitation_energies[3:]) - np.array(excitation_energies[:3])) / dx
        excited_state_forces[i] = lr_force + ground_state_forces[i][:, None, None]

    excited_state_forces = np.transpose(excited_state_forces, (2, 3, 0, 1))
    np.save(os.path.join(dir, "excited_forces.npy"), excited_state_forces)
    
    txt_file = os.path.join(dir, "excited_forces.txt")
    st_labels = ["S", "T"]
    with open(txt_file, "w") as f:
        f.write("# Singlet/Triplet  state_idx  atom_idx  x  y  z\n")
        for st in range(2):
            for istate in range(nstates):
                for iatom in range(natoms):
                    fx, fy, fz = excited_state_forces[st, istate, iatom]
                    f.write(f"{st_labels[st]}  {istate}  {iatom}  {fx:.6f}  {fy:.6f}  {fz:.6f}\n")
    
    return excited_state_forces

def run_diff_all_groundstate(dir=".", abacus_path="abacus", dx=0.001, nproc=1, nparallel=1):
    return run_diff_all_kslr(dir, abacus_path, dx, nproc, nparallel, calculation="scf")

def run_diff_all_lr(dir=".", abacus_path="abacus", dx=0.001, skip_groundstate=False, nproc=1, nparallel=1):
    return run_diff_all_kslr(dir, abacus_path, dx, nproc, nparallel, calculation="scf")

def prepare_diff_custom(
    src_stru, diffed_atom_indices, axes, dx, output_dir="moved_STRU", central=True
):
    """Prepare finite difference STRU files for custom atoms and axes.

    Args:
        src_stru: Path to the source STRU file
        diffed_atom_indices: List of atom indices to displace
        axes: List of axes ('x', 'y', 'z')
        dx: Displacement distance in Angstrom
        output_dir: Output directory for displaced STRU files
        central: Use central difference (default True)
    """
    os.makedirs(output_dir, exist_ok=True)
    src_stru = os.path.abspath(src_stru)
    output_dir = os.path.abspath(output_dir)
    direction_vecs = {}
    if central:
        for axis in axes:
            direction_vecs[axis] = np.array(
                [dx / 2 if ax == axis else 0.0 for ax in ["x", "y", "z"]]
            )
            direction_vecs["-" + axis] = -direction_vecs[axis]
    else:
        for axis in axes:
            direction_vecs[axis] = np.array(
                [dx if ax == axis else 0.0 for ax in ["x", "y", "z"]]
            )
    for i in diffed_atom_indices:
        for axis in axes:
            dr = direction_vecs[axis]
            dest = os.path.join(output_dir, f"STRU_{i}_{axis}")
            move_an_atom_in_stru(src_stru, dest, i, dr)
        if central:
            for axis in axes:
                dr = direction_vecs["-" + axis]
                dest = os.path.join(output_dir, f"STRU_{i}_{'-' + axis}")
                move_an_atom_in_stru(src_stru, dest, i, dr)

def run_diff_custom_groundstate(
    dir=".", abacus_path="abacus", *, diffed_atom_indices, axes, dx=0.001
):
    """Compute ground state forces for custom atoms using finite difference.

    Args:
        dir: Directory containing STRU, INPUT (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        diffed_atom_indices: List of atom indices (0-based) to displace
        axes: List of axes to displace ('x', 'y', 'z')
        dx: Displacement distance in Angstrom (default 0.001)

    Returns:
        forces: dict mapping atom index to axis to force array
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru = os.path.join(dir, "STRU")
    src_input = os.path.join(dir, "INPUT")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")

    prepare_diff_custom(
        src_stru,
        diffed_atom_indices,
        axes,
        dx=dx,
        output_dir=os.path.join(dir, "moved_STRU"),
        central=True,
    )

    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i in diffed_atom_indices:
        for axis in axes + ["-" + axis for axis in axes]:
            task_name = f"STRU_{i}_{axis}"
            task_stru = os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            logger.info(f"Running ABACUS SCF for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="gs.log")

    suffix = grep_parameter_from_input(src_input, "suffix")
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    forces = {}
    for i in diffed_atom_indices:
        forces[i] = {}
        for axis in axes:
            energies = []
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-' + axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                task_scf_log = os.path.join(
                    task_dir, f"OUT.{suffix}", "running_scf.log"
                )
                energy = grep_groundstate_energy_from_log(task_scf_log)
                energies.append(energy)
            force = np.array((energies[1] - energies[0])) / dx
            forces[i][axis] = force
    return forces

def run_diff_custom_lr(
    dir=".",
    abacus_path="abacus",
    *,
    diffed_atom_indices,
    axes,
    dx=0.001,
    skip_groundstate=False,
):
    """Compute linear response TDDFT excited state forces for custom atoms.

    Args:
        dir: Directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        diffed_atom_indices: List of atom indices (0-based) to displace
        axes: List of axes to displace ('x', 'y', 'z')
        dx: Displacement distance in Angstrom (default 0.001)
        skip_groundstate: Skip ground state calculation if already done (default False)

    Returns:
        excited_state_forces: dict mapping atom index to axis to force array
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT_lr")), "INPUT_lr file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"

    src_stru = os.path.join(dir, "STRU")
    src_input_lr = os.path.join(dir, "INPUT_lr")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")
    atoms = _read_abacus(src_stru)
    natoms = len(atoms)
    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    suffix = grep_parameter_from_input(src_input_lr, "suffix")
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    nstates = int(grep_parameter_from_input(src_input_lr, "lr_nstates"))
    if nstates is None:
        nstates = 1

    ground_state_forces = {}
    if skip_groundstate:
        for i in diffed_atom_indices:
            ground_state_forces[i] = {}
            for axis in axes:
                energies = []
                for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-' + axis}"]:
                    task_dir = os.path.join(points_dir, task_name)
                    task_scf_log = os.path.join(
                        task_dir, f"OUT.{suffix}", "running_scf.log"
                    )
                    energy = grep_groundstate_energy_from_log(task_scf_log)
                    energies.append(energy)
                force = np.array((energies[1] - energies[0])) / dx
                ground_state_forces[i][axis] = force
    else:
        os.system(f"cp {os.path.join(dir, 'INPUT_gs')} {os.path.join(dir, 'INPUT')}")
        ground_state_forces = run_diff_custom_groundstate(
            dir, abacus_path, diffed_atom_indices=diffed_atom_indices, axes=axes, dx=dx
        )
        os.system(f"rm {os.path.join(dir, 'INPUT')}")

    for i in diffed_atom_indices:
        for axis in axes + ["-" + axis for axis in axes]:
            task_name = f"STRU_{i}_{axis}"
            task_stru = os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input_lr} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            logger.info(f"Running ABACUS LR for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="lr.log")

    excited_state_forces = {}
    for i in diffed_atom_indices:
        for axis in axes:
            energies = []
            excited_state_forces[i] = {}
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-' + axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                energy = grep_excitation_energy_from_log(
                    os.path.join(task_dir, f"OUT.{suffix}", "running_nscf.log"), nstates
                )
                energies.append(energy)
            lr_force = (np.array(energies[1]) - np.array(energies[0])) / dx
            logger.info(
                f"LR forces for atom {i} along {axis} for each singlet state: {lr_force.reshape(2, -1)[0, :]} eV/Å"
            )
            logger.info(
                f"LR forces for atom {i} along {axis} for each triplet state: {lr_force.reshape(2, -1)[1, :]} eV/Å"
            )
            excited_state_forces[i][axis] = lr_force + ground_state_forces[i][axis]
    return excited_state_forces


def run_diff_custom_kslr(
    dir=".", abacus_path="abacus", *, diffed_atom_indices, axes, dx=0.001
):
    """Compute excited state forces using Kohn-Sham linear response for custom atoms.

    Args:
        dir: Directory containing STRU, INPUT (with lr_nstates) (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        diffed_atom_indices: List of atom indices (0-based) to displace
        axes: List of axes to displace ('x', 'y', 'z')
        dx: Displacement distance in Angstrom (default 0.001)

    Returns:
        excited_state_forces: dict mapping atom index to axis to force array
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru = os.path.join(dir, "STRU")
    src_input = os.path.join(dir, "INPUT")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")

    prepare_diff_custom(
        src_stru,
        diffed_atom_indices,
        axes,
        dx=dx,
        output_dir=os.path.join(dir, "moved_STRU"),
        central=True,
    )

    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i in diffed_atom_indices:
        for axis in axes + ["-" + axis for axis in axes]:
            task_name = f"STRU_{i}_{axis}"
            task_stru = os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            logger.info(f"Running ABACUS SCF for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="ks-lr.log")

    suffix = grep_parameter_from_input(src_input, "suffix")
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    nstates = int(grep_parameter_from_input(src_input, "lr_nstates"))
    if nstates is None:
        nstates = 1
    ground_state_forces = {}
    for i in diffed_atom_indices:
        ground_state_forces[i] = {}
        for axis in axes:
            energies = []
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-' + axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                task_scf_log = os.path.join(
                    task_dir, f"OUT.{suffix}", "running_scf.log"
                )
                energy = grep_groundstate_energy_from_log(task_scf_log)
                energies.append(energy)
            force = np.array((energies[1] - energies[0])) / dx
            ground_state_forces[i][axis] = force
    logger.info(
        "Ground state forces (from finite difference) for each atom: \n",
        ground_state_forces,
    )

    excited_state_forces = {}
    for i in diffed_atom_indices:
        for axis in axes:
            energies = []
            excited_state_forces[i] = {}
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-' + axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                energy = grep_excitation_energy_from_log(
                    os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log"), nstates
                )
                energies.append(energy)
            lr_force = (np.array(energies[1]) - np.array(energies[0])) / dx
            logger.info(
                f"LR forces for atom {i} along {axis} for each singlet state: {lr_force.reshape(2, -1)[0, :]} eV/Å"
            )
            logger.info(
                f"LR forces for atom {i} along {axis} for each triplet state: {lr_force.reshape(2, -1)[1, :]} eV/Å"
            )
            excited_state_forces[i][axis] = lr_force + ground_state_forces[i][axis]
    return excited_state_forces

