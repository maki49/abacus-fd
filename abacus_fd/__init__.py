import argparse
import logging
import sys
import os
from .core import (
    run_diff_all_groundstate,
    run_diff_all_lr,
    run_diff_all_kslr,
    run_diff_custom_groundstate,
    run_diff_custom_lr,
    run_diff_custom_kslr,
    run_single_kslr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

COMMAND_DOCS = {
    "gs-all": {
        "description": "Compute ground state forces for all atoms using finite difference",
        "args": {
            "dir": "Working directory containing STRU, INPUT (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
            "--nproc": "Number of MPI processes per task (default: 1)",
            "--nparallel": "Number of concurrent tasks (default: 1)",
        },
    },
    "lr-all": {
        "description": "Compute LR-TDDFT excited state forces for all atoms",
        "args": {
            "dir": "Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
            "--skip-gs": "Skip ground state calculation if already done (default: False)",
            "--nproc": "Number of MPI processes per task (default: 1)",
            "--nparallel": "Number of concurrent tasks (default: 1)",
        },
    },
    "kslr-all": {
        "description": "Compute excited state forces using Kohn-Sham DFT + LR-TDDFT for all atoms",
        "args": {
            "dir": "Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
            "--nproc": "Number of MPI processes per task (default: 1)",
            "--nparallel": "Number of concurrent tasks (default: 1)",
        },
    },
    "kslr-states": {
        "description": "Run a single SCF + LR-TDDFT calculation to get wavefunctions and amplitudes",
        "args": {
            "dir": "Working directory containing STRU, INPUT (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--nproc": "Number of MPI processes to use (default: 1)",
        },
    },
    "gs-custom": {
        "description": "Compute ground state forces for specified atoms using finite difference",
        "args": {
            "dir": "Working directory containing STRU, INPUT (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--indices": "Comma-separated list of atom indices (0-based) to displace (required)",
            "--axes": "Comma-separated list of axes: x,y,z (required)",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
        },
    },
    "lr-custom": {
        "description": "Compute LR-TDDFT excited state forces for specified atoms",
        "args": {
            "dir": "Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--indices": "Comma-separated list of atom indices (0-based) to displace (required)",
            "--axes": "Comma-separated list of axes: x,y,z (required)",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
            "--skip-gs": "Skip ground state calculation if already done (default: False)",
        },
    },
    "kslr-custom": {
        "description": "Compute excited state forces using Kohn-Sham DFT + LR-TDDFT for specified atoms",
        "args": {
            "dir": "Working directory containing STRU, INPUT with lr_nstates (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--indices": "Comma-separated list of atom indices (0-based) to displace (required)",
            "--axes": "Comma-separated list of axes: x,y,z (required)",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
        },
    },
}


def main():
    parser = argparse.ArgumentParser(
        prog="abacus-fd",
        description="ABACUS Finite Difference: Compute forces via finite difference with ABACUS DFT software",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Helper for common args
    def add_common_args(p):
        p.add_argument("-d", dest="dir", nargs="?", help="Working directory")
        p.add_argument("-a", "--abacus", default="abacus", help="Path to ABACUS binary")
        p.add_argument("-x", "--dx", type=float, default=0.001, help="Displacement distance")
        p.add_argument("-n", "--nproc", type=int, default=1, help="MPI ranks per task")
        p.add_argument("-j", "--nparallel", type=int, default=1, help="Concurrent tasks")

    p = subparsers.add_parser("gs-all", help=COMMAND_DOCS["gs-all"]["description"])
    add_common_args(p)

    p = subparsers.add_parser("lr-all", help=COMMAND_DOCS["lr-all"]["description"])
    add_common_args(p)
    p.add_argument("-s", "--skip-gs", action="store_true", help="Skip ground state calculation")

    p = subparsers.add_parser("kslr-all", help=COMMAND_DOCS["kslr-all"]["description"])
    add_common_args(p)

    p = subparsers.add_parser("kslr-states", help=COMMAND_DOCS["kslr-states"]["description"])
    p.add_argument("-d", dest="dir", nargs="?", help="Working directory")
    p.add_argument("-a", "--abacus", default="abacus", help="Path to ABACUS binary")
    p.add_argument("-n", "--nproc", type=int, default=1, help="Number of MPI processes")

    p = subparsers.add_parser("gs-custom", help=COMMAND_DOCS["gs-custom"]["description"])
    p.add_argument("-d", dest="dir", nargs="?", help="Working directory")
    p.add_argument("-a", "--abacus", default="abacus", help="Path to ABACUS binary")
    p.add_argument("-i", "--indices", required=True, help="Comma-separated atom indices (0-based)")
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument("-x", "--dx", type=float, default=0.001, help="Displacement distance")

    p = subparsers.add_parser("lr-custom", help=COMMAND_DOCS["lr-custom"]["description"])
    p.add_argument("-d", dest="dir", nargs="?", help="Working directory")
    p.add_argument("-a", "--abacus", default="abacus", help="Path to ABACUS binary")
    p.add_argument("-i", "--indices", required=True, help="Comma-separated atom indices (0-based)")
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument("-x", "--dx", type=float, default=0.001, help="Displacement distance")
    p.add_argument("-s", "--skip-gs", action="store_true", help="Skip ground state calculation")

    p = subparsers.add_parser("kslr-custom", help=COMMAND_DOCS["kslr-custom"]["description"])
    p.add_argument("-d", dest="dir", nargs="?", help="Working directory")
    p.add_argument("-a", "--abacus", default="abacus", help="Path to ABACUS binary")
    p.add_argument("-i", "--indices", required=True, help="Comma-separated atom indices (0-based)")
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument("-x", "--dx", type=float, default=0.001, help="Displacement distance")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "gs-all":
        forces = run_diff_all_groundstate(
            dir=args.dir, abacus_path=args.abacus, dx=args.dx, nproc=args.nproc, nparallel=args.nparallel
        )
        logger.info("Forces (eV/Angstrom):\n%s", forces)

    elif args.command == "lr-all":
        forces = run_diff_all_lr(
            dir=args.dir, abacus_path=args.abacus, dx=args.dx, skip_groundstate=args.skip_gs,
            nproc=args.nproc, nparallel=args.nparallel
        )
        logger.info("Excited state forces:\n%s", forces)

    elif args.command == "kslr-all":
        forces = run_diff_all_kslr(
            dir=args.dir, abacus_path=args.abacus, dx=args.dx, nproc=args.nproc, nparallel=args.nparallel
        )
        logger.info("Excited state forces:\n%s", forces)

    elif args.command == "kslr-states":
        run_single_kslr(dir=args.dir, abacus_path=args.abacus, nproc=args.nproc)

    elif args.command == "gs-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_groundstate(
            args.dir, args.abacus, diffed_atom_indices=indices, axes=axes, dx=args.dx
        )
        logger.info("Forces (eV/Angstrom):\n%s", forces)

    elif args.command == "lr-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_lr(
            args.dir, args.abacus, diffed_atom_indices=indices, axes=axes, dx=args.dx, skip_groundstate=args.skip_gs
        )
        logger.info("Excited state forces:\n%s", forces)

    elif args.command == "kslr-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_kslr(
            args.dir, args.abacus, diffed_atom_indices=indices, axes=axes, dx=args.dx
        )
        logger.info("Excited state forces:\n%s", forces)


if __name__ == "__main__":
    main()
