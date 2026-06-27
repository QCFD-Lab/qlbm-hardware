import numpy as np
from pathlib import Path
from typing import Tuple, List
from qiskit.quantum_info import Statevector


def load_statevector(path_to_statevector: str) -> Statevector:
    """
    Load and return statevector from a .npy file as a Qiskit Statevector object.

    Args:
        path_to_statevector: Path to numpy statevector file

    Returns:
        Statevector object from Qiskit
    """
    path = Path(path_to_statevector)
    if not path.exists():
        raise FileNotFoundError(f"Statevector file not found: {path_to_statevector}")
    if not path.suffix == ".npy":
        raise ValueError(f"Invalid file format: {path_to_statevector}. Expected .npy file.")

    vec = np.load(path)
    return Statevector(vec)


def compare_statevectors(dir_unoptimized: str, dir_optimized: str, tolerance: float = 1e-6) -> dict:
    """
    Compare statevector files between optimized and unoptimized simulation outputs.

    Internally uses compare_statevectors_ignoring_global_phase() for each file pair.

    Args:
        dir_unoptimized: Path to unoptimized simulation output directory
        dir_optimized: Path to optimized simulation output directory
        tolerance: Numerical tolerance for comparison (default: 1e-6)

    Returns:
        dict with:
        - all_match: True if all statevectors are equivalent within tolerance
        - differences: List of strings describing any mismatches found
        - file_results: List of detailed comparison dicts for each file pair
    """
    statevectors_dir_unopt = Path(dir_unoptimized) / "statevectors"
    statevectors_dir_opt = Path(dir_optimized) / "statevectors"

    if not statevectors_dir_unopt.exists():
        return {
            "all_match": False,
            "differences": [f"Unoptimized statevectors directory not found: {statevectors_dir_unopt}"],
            "file_results": [],
        }
    if not statevectors_dir_opt.exists():
        return {
            "all_match": False,
            "differences": [f"Optimized statevectors directory not found: {statevectors_dir_opt}"],
            "file_results": [],
        }

    files_unopt = sorted(statevectors_dir_unopt.glob("step_*.npy"))
    files_opt = sorted(statevectors_dir_opt.glob("step_*.npy"))

    if len(files_unopt) != len(files_opt):
        return {
            "all_match": False,
            "differences": [
                f"Number of statevector files mismatch: "
                f"Unoptimized={len(files_unopt)}, Optimized={len(files_opt)}"
            ],
            "file_results": [],
        }

    differences = []
    all_match = True
    file_results = []

    for file_unopt, file_opt in zip(files_unopt, files_opt):
        step_name = file_unopt.name
        try:
            result = compare_statevectors_ignoring_global_phase(
                file_unopt, file_opt, tolerance=tolerance
            )
            file_results.append(result)
        except Exception as e:
            differences.append(f"Failed to compare {step_name}: {e}")
            all_match = False
            file_results.append({"error": str(e)})
            continue

        if not result["statevectors_equal_ignoring_phase"]:
            differences.append(
                f"{step_name}: Not equivalent - "
                f"Max absolute difference={result['max_absolute_difference']:.6e}, "
                f"Overlap={result['overlap_abs']:.6e}"
            )
            all_match = False

    return {
        "all_match": all_match,
        "differences": differences,
        "file_results": file_results,
    }


def compare_statevectors_ignoring_global_phase(file1_path, file2_path, tolerance=1e-10):
    state1 = np.load(file1_path)
    state2 = np.load(file2_path)

    norm1 = np.sum(np.abs(state1) ** 2)
    norm2 = np.sum(np.abs(state2) ** 2)

    overlap = np.vdot(state1, state2)  # <state1|state2>
    overlap_abs = np.abs(overlap)

    if overlap_abs < tolerance:
        # If overlap is ~0, a global phase alignment is not meaningful.
        state2_phase_removed = state2.copy()
        global_phase = None
        is_equal = False
    else:
        phase = overlap / overlap_abs
        state2_phase_removed = state2 / phase
        global_phase = np.angle(phase)
        is_equal = np.allclose(state1, state2_phase_removed, atol=tolerance)

    absolute_diff = np.abs(state1 - state2_phase_removed)
    relative_diff = absolute_diff / (np.abs(state1) + 1e-12)
    diff_indices = np.where(absolute_diff > tolerance)[0]

    return {
        "state1_length": len(state1),
        "state2_length": len(state2),
        "state1_norm": norm1,
        "state2_norm": norm2,
        "overlap_abs": overlap_abs,
        "global_phase_diff": global_phase,
        "statevectors_equal_ignoring_phase": is_equal,
        "absolute_differences": absolute_diff,
        "relative_differences": relative_diff,
        "indices_with_large_differences": diff_indices,
        "max_absolute_difference": np.max(absolute_diff),
    }