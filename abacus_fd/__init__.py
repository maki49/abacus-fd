import argparse
import sys
from .core import (
    run_diff_all_groundstate,
    run_diff_all_lr,
    run_diff_all_kslr,
    run_diff_custom_groundstate,
    run_diff_custom_lr,
    run_diff_custom_kslr,
)

COMMAND_DOCS = {
    "gs-all": {
        "description": "Compute ground state forces for all atoms using finite difference",
        "args": {
            "dir": "Working directory containing STRU, INPUT (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
        },
    },
    "lr-all": {
        "description": "Compute LR-TDDFT excited state forces for all atoms",
        "args": {
            "dir": "Working directory containing STRU, INPUT_gs, INPUT_lr (and optionally KPT) files",
            "abacus": "Path to the ABACUS executable",
            "--dx": "Displacement distance in Angstrom (default: 0.001)",
            "--skip-gs": "Skip ground state calculation if already done (default: False)",
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
        "description": "Compute excited state forces using Kohn-Sham LR for specified atoms",
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

    p = subparsers.add_parser("gs-all", help=COMMAND_DOCS["gs-all"]["description"])
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )

    p = subparsers.add_parser("lr-all", help=COMMAND_DOCS["lr-all"]["description"])
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )
    p.add_argument(
        "--skip-gs", action="store_true", help="Skip ground state calculation"
    )

    p = subparsers.add_parser(
        "gs-custom", help=COMMAND_DOCS["gs-custom"]["description"]
    )
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--indices", required=True, help="Comma-separated atom indices (0-based)"
    )
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )

    p = subparsers.add_parser(
        "lr-custom", help=COMMAND_DOCS["lr-custom"]["description"]
    )
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--indices", required=True, help="Comma-separated atom indices (0-based)"
    )
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )
    p.add_argument(
        "--skip-gs", action="store_true", help="Skip ground state calculation"
    )

    p = subparsers.add_parser(
        "kslr-custom", help=COMMAND_DOCS["kslr-custom"]["description"]
    )
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--indices", required=True, help="Comma-separated atom indices (0-based)"
    )
    p.add_argument("--axes", required=True, help="Comma-separated axes (x,y,z)")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )

    p = subparsers.add_parser("kslr-all", help=COMMAND_DOCS["kslr-all"]["description"])
    p.add_argument("dir", help="Working directory")
    p.add_argument("abacus", help="Path to ABACUS executable")
    p.add_argument(
        "--dx", type=float, default=0.001, help="Displacement distance (default: 0.001)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "gs-all":
        forces = run_diff_all_groundstate(args.dir, args.abacus, args.dx)
        print("Forces (eV/Angstrom):", forces)

    elif args.command == "lr-all":
        forces = run_diff_all_lr(args.dir, args.abacus, args.dx, args.skip_gs)
        print("Excited state forces:", forces)

    elif args.command == "gs-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_groundstate(
            args.dir, args.abacus, indices, axes, args.dx
        )
        print("Forces (eV/Angstrom):", forces)

    elif args.command == "lr-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_lr(
            args.dir, args.abacus, indices, axes, args.dx, args.skip_gs
        )
        print("Excited state forces:", forces)

    elif args.command == "kslr-custom":
        indices = [int(x.strip()) for x in args.indices.split(",")]
        axes = [x.strip() for x in args.axes.split(",")]
        forces = run_diff_custom_kslr(args.dir, args.abacus, indices, axes, args.dx)
        print("Excited state forces:", forces)

    elif args.command == "kslr-all":
        forces = run_diff_all_kslr(args.dir, args.abacus, args.dx)
        print("Excited state forces:", forces)


if __name__ == "__main__":
    main()
