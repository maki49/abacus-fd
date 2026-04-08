from ase.io import read, write
import numpy as np
import os, subprocess
import logging

logger = logging.getLogger(__name__)


def grep_parameter_from_input(input_file, para_name):
    """grep parameter from abacus INPUT file"""
    command = f"grep {para_name} {input_file} | awk '{{print $2}}'"
    text = subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    if text == "":
        return None
    return text.strip()


def grep_groundstate_energy_from_log(log):
    """grep groundstate energy (eV) from abacus log file"""
    command = f"grep '!FINAL_ETOT_IS' {log} | awk '{{print $2}}'"
    text = (
        subprocess.run(command, shell=True, capture_output=True)
        .stdout.decode()
        .strip()
        .strip("\n")
    )
    try:
        e = float(text)
    except ValueError:
        command = f"grep 'E_KohnSham' {log} | tail -n 1 | awk '{{print $3}}'"
        text = (
            subprocess.run(command, shell=True, capture_output=True)
            .stdout.decode()
            .strip()
            .strip("\n")
        )
        try:
            e = float(text)
        except ValueError:
            raise ValueError(
                f"Cannot parse energy from log file {log}. Please check the log file for errors."
            )
    return e


def grep_excitation_energy_from_log(log, nstate):
    """grep LR-TDDFT excitation energy (eV) from abacus log file"""
    command = f"grep -A {nstate + 1} 'Excitation Energy' {log} | grep [0-9] | awk '{{print $3}}'"
    text = subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    return np.array([float(x) for x in text.split()]).reshape(2, -1).tolist()


def grep_excitation_energy_from_output(outfile):
    """grep LR-TDDFT excitation energy (eV) from abacus output file"""
    command = f"grep -A 1 'eigenvalues: (eV)' {outfile} | grep [0-9] "
    text = subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    return np.array([float(x) for x in text.split()]).reshape(2, -1).tolist()


def move_an_atom_in_stru(src, dest, atom_index, dr, scaled=False):
    """Move an atom in the `src` STRU file by a specified distance (in Angstrom), and write the modified STRU file to `dest`."""
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    atoms = read(src, format="abacus")
    positions = atoms.get_positions()
    positions[atom_index] += np.array(dr)
    atoms.set_positions(positions)

    write(dest, atoms, format="abacus", scaled=scaled)


def prepare_diff_all(src_stru, dx, output_dir="moved_STRU", central=True):
    """Prepare finite difference STRU files by moving each atom in the `src_stru` file by a specified distance `dx` along x, y, and z directions.

    Args:
        src_stru: Path to the source STRU file
        dx: Displacement distance in Angstrom
        output_dir: Output directory for displaced STRU files
        central: Use central difference (default True)
    """
    os.makedirs(output_dir, exist_ok=True)
    src_stru = os.path.abspath(src_stru)
    output_dir = os.path.abspath(output_dir)
    atoms = read(src_stru, format="abacus")
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


def run_abacus(dir, abacus, nproc=1, log="log.txt"):
    cwd = os.getcwd()
    absdir = os.path.abspath(dir)
    os.system(
        f"cd {absdir} && mpirun -np {nproc} {os.path.abspath(abacus)} > {log} && cd {cwd}"
    )


def run_diff_all_groundstate(dir=".", abacus_path="abacus", dx=0.001):
    """Compute ground state forces for all atoms using finite difference.

    Args:
        dir: Directory containing STRU, INPUT (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        dx: Displacement distance in Angstrom (default 0.001)

    Returns:
        forces: numpy array of shape (natoms, 3) containing forces in eV/Angstrom
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru = os.path.join(dir, "STRU")
    src_input = os.path.join(dir, "INPUT")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")

    prepare_diff_all(
        src_stru=src_stru,
        dx=dx,
        output_dir=os.path.join(dir, "moved_STRU"),
        central=True,
    )
    atoms = read(src_stru, format="abacus")
    natoms = len(atoms)

    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i in range(natoms):
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
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
    forces = np.zeros((natoms, 3))
    for i in range(natoms):
        energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
            energy = grep_groundstate_energy_from_log(task_scf_log)
            energies.append(energy)
        force = (np.array((energies[3:]) - np.array(energies[:3]))) / dx
        forces[i] = force
    return forces


def run_diff_all_lr(dir=".", abacus_path="abacus", dx=0.001, skip_groundstate=False):
    """Compute linear response TDDFT excited state forces for all atoms.

    Args:
        dir: Directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        dx: Displacement distance in Angstrom (default 0.001)
        skip_groundstate: Skip ground state calculation if already done (default False)

    Returns:
        excited_state_forces: numpy array of shape (2, nstates, natoms, 3)
                          [0] = singlet forces, [1] = triplet forces
                          in eV/Angstrom.
        Output files: excited_forces.npy, excited_forces.txt
        TXT format (columns): Singlet/Triplet  state_idx  atom_idx  x  y  z
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT_lr")), "INPUT_lr file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"

    src_stru = os.path.join(dir, "STRU")
    src_input_lr = os.path.join(dir, "INPUT_lr")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")
    atoms = read(src_stru, format="abacus")
    natoms = len(atoms)
    points_dir = os.path.join(dir, "points")
    if not os.path.exists(points_dir):
        skip_groundstate = False

    suffix = grep_parameter_from_input(src_input_lr, "suffix")
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    nstates = int(grep_parameter_from_input(src_input_lr, "lr_nstates"))
    if nstates is None:
        nstates = 1

    ground_state_forces = np.zeros((natoms, 3))
    if skip_groundstate:
        for i in range(natoms):
            energies = []
            for axis in ["x", "y", "z", "-x", "-y", "-z"]:
                task_name = f"STRU_{i}_{axis}"
                task_dir = os.path.join(points_dir, task_name)
                task_scf_log = os.path.join(
                    task_dir, f"OUT.{suffix}", "running_scf.log"
                )
                energy = grep_groundstate_energy_from_log(task_scf_log)
                energies.append(energy)
            force = (np.array((energies[3:]) - np.array(energies[:3]))) / dx
            ground_state_forces[i] = force
    else:
        os.system(f"cp {os.path.join(dir, 'INPUT_gs')} {os.path.join(dir, 'INPUT')}")
        ground_state_forces = run_diff_all_groundstate(dir, abacus_path, dx)
        os.system(f"rm {os.path.join(dir, 'INPUT')}")
    logger.info(
        "Ground state forces (from finite difference) for each atom: \n",
        ground_state_forces,
    )

    for i in range(natoms):
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_stru = os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input_lr} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            run_abacus(task_dir, abacus_path, log="lr.log")

    excited_state_forces = np.zeros((natoms, 3, 2, nstates))
    for i in range(natoms):
        excitation_energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            energy = grep_excitation_energy_from_log(
                os.path.join(task_dir, f"OUT.{suffix}", "running_nscf.log"), nstates
            )
            excitation_energies.append(energy)
        excitation_energies = np.array(excitation_energies).reshape(6, 2, nstates)
        lr_force = (excitation_energies[3:, :, :] - excitation_energies[:3, :, :]) / dx
        logger.info(
            f"LR forces for atom {i} along x for each singlet state: {lr_force[0, :, :].reshape(2, -1)[0, :]} eV/Å"
        )
        logger.info(
            f"LR forces for atom {i} along y for each triplet state: {lr_force[1, :, :].reshape(2, -1)[1, :]} eV/Å"
        )
        logger.info(
            f"LR forces for atom {i} along z for each singlet state: {lr_force[2, :, :].reshape(2, -1)[0, :]} eV/Å"
        )
        excited_state_forces[i] = lr_force + ground_state_forces[i][:, None, None]

    excited_state_forces = np.transpose(excited_state_forces, (2, 3, 0, 1))

    npy_file = os.path.join(dir, "excited_forces.npy")
    np.save(npy_file, excited_state_forces)
    logger.info(f"Saved forces to {npy_file}")

    txt_file = os.path.join(dir, "excited_forces.txt")
    st_labels = ["S", "T"]
    with open(txt_file, "w") as f:
        f.write("# Singlet/Triplet  state_idx  atom_idx  x  y  z\n")
        for st in range(2):
            for istate in range(nstates):
                for iatom in range(natoms):
                    fx, fy, fz = excited_state_forces[st, istate, iatom]
                    f.write(
                        f"{st_labels[st]}  {istate}  {iatom}  {fx:.6f}  {fy:.6f}  {fz:.6f}\n"
                    )
    logger.info(f"Saved forces to {txt_file}")

    logger.info(
        "Excited state forces (from finite difference) for each atom: \n",
        excited_state_forces,
    )
    return excited_state_forces


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
    atoms = read(src_stru, format="abacus")
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
            dir, abacus_path, diffed_atom_indices, axes, dx
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


def run_diff_all_kslr(dir=".", abacus_path="abacus", dx=0.001):
    """Compute excited state forces using Kohn-Sham linear response for all atoms.

    Args:
        dir: Directory containing STRU, INPUT (with lr_nstates) (and optionally KPT) files
        abacus_path: Path to the ABACUS executable
        dx: Displacement distance in Angstrom (default 0.001)

    Returns:
        excited_state_forces: numpy array of shape (2, nstates, natoms, 3)
                          [0] = singlet forces, [1] = triplet forces
                          in eV/Angstrom.
        Output files: excited_forces.npy, excited_forces.txt
        TXT format (columns): Singlet/Triplet  state_idx  atom_idx  x  y  z
    """
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru = os.path.join(dir, "STRU")
    src_input = os.path.join(dir, "INPUT")
    src_kpt = None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt = os.path.join(dir, "KPT")

    prepare_diff_all(
        src_stru=src_stru,
        dx=dx,
        output_dir=os.path.join(dir, "moved_STRU"),
        central=True,
    )
    atoms = read(src_stru, format="abacus")
    natoms = len(atoms)

    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i in range(natoms):
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
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
    ground_state_forces = np.zeros((natoms, 3))
    for i in range(natoms):
        energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
            energy = grep_groundstate_energy_from_log(task_scf_log)
            energies.append(energy)
        force = (np.array((energies[3:]) - np.array(energies[:3]))) / dx
        ground_state_forces[i] = force

    excited_state_forces = np.zeros((natoms, 3, 2, nstates))
    for i in range(natoms):
        excitation_energies = []
        for axis in ["x", "y", "z", "-x", "-y", "-z"]:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            energy = grep_excitation_energy_from_log(
                os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log"), nstates
            )
            excitation_energies.append(energy)
        excitation_energies = np.array(excitation_energies).reshape(6, 2, nstates)
        lr_force = (excitation_energies[3:, :, :] - excitation_energies[:3, :, :]) / dx
        logger.info(
            f"LR forces for atom {i} along x for each singlet state: {lr_force[0, :, :].reshape(2, -1)[0, :]} eV/Å"
        )
        logger.info(
            f"LR forces for atom {i} along y for each triplet state: {lr_force[1, :, :].reshape(2, -1)[1, :]} eV/Å"
        )
        logger.info(
            f"LR forces for atom {i} along z for each singlet state: {lr_force[2, :, :].reshape(2, -1)[0, :]} eV/Å"
        )
        excited_state_forces[i] = lr_force + ground_state_forces[i][:, None, None]

    excited_state_forces = np.transpose(excited_state_forces, (2, 3, 0, 1))

    npy_file = os.path.join(dir, "excited_forces.npy")
    np.save(npy_file, excited_state_forces)
    logger.info(f"Saved forces to {npy_file}")

    txt_file = os.path.join(dir, "excited_forces.txt")
    st_labels = ["S", "T"]
    with open(txt_file, "w") as f:
        f.write("# Singlet/Triplet  state_idx  atom_idx  x  y  z\n")
        for st in range(2):
            for istate in range(nstates):
                for iatom in range(natoms):
                    fx, fy, fz = excited_state_forces[st, istate, iatom]
                    f.write(
                        f"{st_labels[st]}  {istate}  {iatom}  {fx:.6f}  {fy:.6f}  {fz:.6f}\n"
                    )
    logger.info(f"Saved forces to {txt_file}")

    logger.info(
        "Excited state forces (from finite difference) for each atom: \n",
        excited_state_forces,
    )
    return excited_state_forces
