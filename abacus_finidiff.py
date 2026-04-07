from ase.io import read, write
import numpy as np
import os, subprocess

def grep_parameter_from_input(input_file, para_name):
    """grep parameter from abacus INPUT file"""
    command=f"grep {para_name} {input_file} | awk '{{print $2}}'"
    text=subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    if text == "":
        return None
    return text.strip()

def grep_groundstate_energy_from_log(log):
    """grep groundstate energy (eV) from abacus log file"""
    command=f"grep '!FINAL_ETOT_IS' {log} | awk '{{print $2}}'"
    text=subprocess.run(command, shell=True, capture_output=True).stdout.decode().strip().strip('\n')
    return float(text)

def grep_excitation_energy_from_log(log, nstate):
    """grep LR-TDDFT excitation energy (eV) from abacus log file"""
    command=f"grep -A {nstate+1} 'Excitation Energy' {log} | grep [0-9] | awk '{{print $3}}'"
    text=subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    return np.array([float(x) for x in text.split()]).reshape(2, -1).tolist()   # 2 for singlet and triplet

def grep_excitation_energy_from_output(outfile):
    """grep LR-TDDFT excitation energy (eV) from abacus output file"""
    command=f"grep -A 1 'eigenvalues: (eV)' {outfile} | grep [0-9] "
    text=subprocess.run(command, shell=True, capture_output=True).stdout.decode()
    return np.array([float(x) for x in text.split()]).reshape(2, -1).tolist()   # 2 for singlet and triplet

def move_an_atom_in_stru(src, dest, atom_index, dr, scaled=False):
    """Move an atom in the `src` STRU file by a specified distance (in Angstrom), and write the modified STRU file to `dest`."""
    src=os.path.abspath(src)
    dest=os.path.abspath(dest)
    atoms = read(src, format='abacus')
    positions = atoms.get_positions()
    positions[atom_index] += np.array(dr)
    atoms.set_positions(positions)
    
    write(dest, atoms, format='abacus', scaled=scaled)


def prepare_diff_all(src_stru, dx,output_dir="moved_STRU", central=True):
    """ Prepare finite difference STRU files by moving each atom in the `src_stru` file by a specified distance `dx` along x, y, and z directions, and saving the modified STRU files in the `output_dir` directory.

    Args:
        src_stru (_type_): _description_
        dx (_type_): _description_
        output_dir (str, optional): _description_. Defaults to "moved_STRU".
    """
    os.makedirs(output_dir, exist_ok=True)
    src_stru=os.path.abspath(src_stru)
    output_dir=os.path.abspath(output_dir)
    atoms = read(src_stru, format='abacus')
    natoms = len(atoms)
    if central:
        direction_vecs = {
            'x': np.array([dx/2, 0.0, 0.0]),
            'y': np.array([0.0, dx/2, 0.0]),
            'z': np.array([0.0, 0.0, dx/2]),
            '-x': np.array([-dx/2, 0.0, 0.0]),
            '-y': np.array([0.0, -dx/2, 0.0]),
            '-z': np.array([0.0, 0.0, -dx/2]),
        }
    else:
        direction_vecs = {
            'x': np.array([dx, 0.0, 0.0]),
            'y': np.array([0.0, dx, 0.0]),
            'z': np.array([0.0, 0.0, dx]),
        }
    for i in range(natoms):
        for axis, dr in direction_vecs.items():
            dest = os.path.join(output_dir, f"STRU_{i}_{axis}")
            move_an_atom_in_stru(src_stru, dest, i, dr)
            # print(f"Created {dest} by moving atom {i} along {axis} by {dx} Å")
            
def prepare_diff_custom(src_stru, diffed_atom_indices, axes, dx, output_dir="moved_STRU", central=True):
    os.makedirs(output_dir, exist_ok=True)
    src_stru=os.path.abspath(src_stru)
    output_dir=os.path.abspath(output_dir)
    direction_vecs = {}
    if central:
        for axis in axes:
            direction_vecs[axis] = np.array([dx/2 if ax == axis else 0.0 for ax in ['x', 'y', 'z']])
            direction_vecs['-'+axis] = -direction_vecs[axis]
    else:
        for axis in axes:
            direction_vecs[axis] = np.array([dx if ax == axis else 0.0 for ax in ['x', 'y', 'z']])
    for i in diffed_atom_indices:
        for axis in axes:
            dr = direction_vecs[axis]
            dest = os.path.join(output_dir, f"STRU_{i}_{axis}")
            move_an_atom_in_stru(src_stru, dest, i, dr)
        if central:
            for axis in axes:
                dr = direction_vecs['-'+axis]
                dest = os.path.join(output_dir, f"STRU_{i}_{'-'+axis}")
                move_an_atom_in_stru(src_stru, dest, i, dr)

def run_abacus(dir, abacus, nproc=1, log="log.txt"):
    cwd=os.getcwd()
    absdir=os.path.abspath(dir)
    os.system(f"cd {absdir} && mpirun -np {nproc} {os.path.abspath(abacus)} > {log} && cd {cwd}")


def run_diff_all_groundstate(dir, abacus_path, dx=0.001):
    """dir should contain the original STRU, INPUT and KPT (optional) files.
    abacus_dir is the path to the ABACUS executable."""
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru=os.path.join(dir, "STRU")
    src_input=os.path.join(dir, "INPUT")
    src_kpt=None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt=os.path.join(dir, "KPT")
        
    # 1. prepare diffed STRU files
    prepare_diff_all(src_stru=src_stru, dx=dx, output_dir=os.path.join(dir, "moved_STRU"), central=True)
    atoms = read(src_stru, format='abacus')
    natoms = len(atoms)
    
    # 2. run ABACUS for each diffed STRU file
    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)
    forces=[]
    for i in range(natoms):
        for axis in ['x', 'y', 'z', '-x', '-y', '-z']:
            task_name = f"STRU_{i}_{axis}"
            task_stru=os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            print(f"Running ABACUS SCF for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="gs.log")

    # 3. get forces
    suffix = grep_parameter_from_input(src_input, "suffix") 
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    forces = np.zeros((natoms, 3))
    for i in range(natoms):
        energies = []
        for axis in ['x', 'y', 'z', '-x', '-y', '-z']:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
            energy=grep_groundstate_energy_from_log(task_scf_log)
            energies.append(energy)
        force = (np.array((energies[3:]) - np.array(energies[:3])))/dx    # F=-dE/dx
        forces[i] = force
    return forces


def run_diff_all_lr(dir, abacus_path, dx=0.001, skip_groundstate=False):
    """ linear response TDDFT calculation for finite difference.
    dir should contain the original STRU, INPUT_gs, INPUT_lr and KPT (optional) files.
    abacus_dir is the path to the ABACUS executable.
    if groundstates have been calculated int `dir`, set skip_groundstate=True to save time."""
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT_lr")), "INPUT_lr file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    
    src_stru=os.path.join(dir, "STRU")
    src_input_lr=os.path.join(dir, "INPUT_lr")
    src_kpt=None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt=os.path.join(dir, "KPT")
    atoms = read(src_stru, format='abacus')
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
        
    # 1. prepare diffed STRU files and groundstates
    forces_gs = np.zeros((natoms, 3))
    if skip_groundstate:
        for i in range(natoms):
            energies = []
            for axis in ['x', 'y', 'z', '-x', '-y', '-z']:
                task_name = f"STRU_{i}_{axis}"
                task_dir = os.path.join(points_dir, task_name)
                task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
                energy=grep_groundstate_energy_from_log(task_scf_log)
                energies.append(energy)
            force = (np.array((energies[3:]) - np.array(energies[:3])))/dx    # F=-dE/dx
            forces_gs[i] = force
    else: 
        os.system(f"cp {os.path.join(dir, 'INPUT_gs')} {os.path.join(dir, 'INPUT')}")
        forces_gs = run_diff_all_groundstate(dir, abacus_path, dx)
        os.system(f"rm {os.path.join(dir, 'INPUT')}")   # remove the copied INPUT file
    print("Ground state forces (from finite difference) for each atom: \n", forces_gs)

    # 2. run ABACUS LR for each diffed STRU file
    for i in range(natoms):
        for axis in ['x', 'y', 'z', '-x', '-y', '-z']:
            task_name = f"STRU_{i}_{axis}"
            task_stru=os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input_lr} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            # print(f"Running ABACUS LR for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="lr.log")
            
    # 3. get excitation energies
    excited_state_forces = np.zeros((natoms, 3, 2, nstates))   # 2 for singlet and triplet
    for i in range(natoms):
        excitation_energies = []
        for axis in ['x', 'y', 'z', '-x', '-y', '-z']:
            task_name = f"STRU_{i}_{axis}"
            task_dir = os.path.join(points_dir, task_name)
            energy = grep_excitation_energy_from_log(os.path.join(task_dir, f"OUT.{suffix}", "running_nscf.log"), nstates)
            excitation_energies.append(energy)  # len: nstates*2(singlet/triplet)
        excitation_energies = np.array(excitation_energies).reshape(6, 2, nstates)   # 6 for +x, +y, +z, -x, -y, -z
        lr_force = (excitation_energies[3:, :, :] - excitation_energies[:3, :, :])/dx    # -d\Omega/dx
        print(f"LR forces (minus excited states energy gradients) for atom {i} along x for each singlet state: {lr_force[0, :, :].reshape(2,-1)[0,:]} eV/Å")
        print(f"LR forces (minus excited states energy gradients) for atom {i} along y for each triplet state: {lr_force[1, :, :].reshape(2,-1)[1,:]} eV/Å")
        print(f"LR forces (minus excited states energy gradients) for atom {i} along z for each singlet state: {lr_force[2, :, :].reshape(2,-1)[0,:]} eV/Å")
        excited_state_forces[i] = lr_force + forces_gs[i][:, None, None]   # add groundstate forces to get total forces in excited states
    print("Excited state forces (from finite difference) for each atom: \n", excited_state_forces)
    return excited_state_forces


def run_diff_custom_groundstate(dir, abacus_path, diffed_atom_indices, axes, dx=0.001):
    """dir should contain the original STRU, INPUT and KPT (optional) files.
    abacus_dir is the path to the ABACUS executable."""
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT")), "INPUT file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    src_stru=os.path.join(dir, "STRU")
    src_input=os.path.join(dir, "INPUT")
    src_kpt=None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt=os.path.join(dir, "KPT")
        
    # 1. prepare diffed STRU files
    prepare_diff_custom(src_stru, diffed_atom_indices, axes, dx=dx, output_dir=os.path.join(dir, "moved_STRU"), central=True)
    
    # 2. run ABACUS for each diffed STRU file
    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)
    forces=[]
    for i in diffed_atom_indices:
        for axis in axes + ['-'+axis for axis in axes]:
            task_name = f"STRU_{i}_{axis}"
            task_stru=os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            print(f"Running ABACUS SCF for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="gs.log")

    # 3. get forces
    suffix = grep_parameter_from_input(src_input, "suffix") 
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    forces = {}
    for i in diffed_atom_indices:
        forces[i] = {}
        for axis in axes:
            energies = []
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-'+axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
                energy=grep_groundstate_energy_from_log(task_scf_log)
                energies.append(energy)
            force = np.array((energies[1] - energies[0]))/dx    # F=-dE/dx
            forces[i][axis] = force
    return forces


def run_diff_custom_lr(dir, abacus_path,  diffed_atom_indices, axes, dx=0.001, skip_groundstate=False):
    """ linear response TDDFT calculation for finite difference.
    dir should contain the original STRU, INPUT_gs, INPUT_lr and KPT (optional) files.
    abacus_dir is the path to the ABACUS executable.
    if groundstates have been calculated int `dir`, set skip_groundstate=True to save time."""
    dir = os.path.abspath(dir)
    assert os.path.exists(os.path.join(dir, "INPUT_lr")), "INPUT_lr file not found"
    assert os.path.exists(os.path.join(dir, "STRU")), "STRU file not found"
    
    src_stru=os.path.join(dir, "STRU")
    src_input_lr=os.path.join(dir, "INPUT_lr")
    src_kpt=None
    if os.path.exists(os.path.join(dir, "KPT")):
        src_kpt=os.path.join(dir, "KPT")
    atoms = read(src_stru, format='abacus')
    natoms = len(atoms)
    points_dir = os.path.join(dir, "points")
    os.makedirs(points_dir, exist_ok=True)
    # if not os.path.exists(points_dir):
    #     skip_groundstate = False
        
    suffix = grep_parameter_from_input(src_input_lr, "suffix") 
    if suffix is None or suffix == "":
        suffix = "ABACUS"
    nstates = int(grep_parameter_from_input(src_input_lr, "lr_nstates"))
    if nstates is None:
        nstates = 1
        
    # 1. prepare diffed STRU files and groundstates
    ground_state_forces = {}
    if skip_groundstate:
        for i in diffed_atom_indices:
            ground_state_forces[i] = {}
            for axis in axes:
                energies = []
                for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-'+axis}"]:
                    task_dir = os.path.join(points_dir, task_name)
                    task_scf_log = os.path.join(task_dir, f"OUT.{suffix}", "running_scf.log")
                    energy=grep_groundstate_energy_from_log(task_scf_log)
                    energies.append(energy)
                force = np.array((energies[1] - energies[0]))/dx    # F=-dE/dx
                ground_state_forces[i][axis] = force
    else: 
        os.system(f"cp {os.path.join(dir, 'INPUT_gs')} {os.path.join(dir, 'INPUT')}")
        ground_state_forces = run_diff_custom_groundstate(dir, abacus_path, diffed_atom_indices, axes, dx)
        os.system(f"rm {os.path.join(dir, 'INPUT')}")   # remove the copied INPUT file

    # 2. run ABACUS LR for each diffed STRU file
    for i in diffed_atom_indices:
        for axis in axes + ['-'+axis for axis in axes]:
            task_name = f"STRU_{i}_{axis}"
            task_stru=os.path.join(dir, "moved_STRU", task_name)
            task_dir = os.path.join(points_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            os.system(f"cp {src_input_lr} {os.path.join(task_dir, 'INPUT')}")
            os.system(f"cp {task_stru} {os.path.join(task_dir, 'STRU')}")
            if src_kpt is not None:
                os.system(f"cp {src_kpt} {os.path.join(task_dir, 'KPT')}")
            print(f"Running ABACUS LR for atom {i} moved along {axis}...")
            run_abacus(task_dir, abacus_path, log="lr.log")
            
    # 3. get excitation energies
    excited_state_forces = {}
    for i in diffed_atom_indices:
        for axis in axes:
            energies = []
            excited_state_forces[i] = {}
            for task_name in [f"STRU_{i}_{axis}", f"STRU_{i}_{'-'+axis}"]:
                task_dir = os.path.join(points_dir, task_name)
                energy = grep_excitation_energy_from_log(os.path.join(task_dir, f"OUT.{suffix}", "running_nscf.log"), nstates)
                energies.append(energy) # len: nstates*2(singlet/triplet)
            lr_force = (np.array(energies[1]) - np.array(energies[0]))/dx    # -d\Omega/dx
            print(f"LR forces (minus excited states energy gradients) for atom {i} along {axis} for each singlet state: {lr_force.reshape(2,-1)[0,:]} eV/Å")
            print(f"LR forces (minus excited states energy gradients) for atom {i} along {axis} for each triplet state: {lr_force.reshape(2,-1)[1,:]} eV/Å")
            excited_state_forces[i][axis] = lr_force + ground_state_forces[i][axis]   # add groundstate forces to get total forces in excited states
    return excited_state_forces

if __name__ == "__main__":

    # abacus_path = "/home/fortneu49/LR-Grad/abacus-develop/build/abacus" # LTS
    abacus_path = "/home/fortneu49/abacus-fix/abacus-develop/build/abacus_3p"   #develop
    
    # For a single atom
    forces = run_diff_custom_lr(dir=".", abacus_path=abacus_path, diffed_atom_indices=[1], axes=['z'], dx=0.001, skip_groundstate=False)
    
    # For all atoms
    # forces = run_diff_all_lr(dir=".", abacus_path=abacus_path, dx=0.001, skip_groundstate=False)
    
    print("forces = ", forces)