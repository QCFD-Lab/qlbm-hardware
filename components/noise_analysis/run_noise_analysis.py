"""Choose and run one QLBM hardware noise-analysis experiment."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"
for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.noise_analysis.density_experiments import (
    run_density_error_growth_comparison,
    run_depolarizing_probability_sweep,
    run_multi_hardware_density_depolarizing_comparison,
    run_noiseless_vs_noise_density_comparison,
)
from components.noise_analysis.experiment_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
)
from components.noise_analysis.generic_experiments import (
    run_hardware_noise_analysis,
)
from components.noise_analysis.velocity_experiments import (
    run_multi_hardware_velocity_profile_comparison,
    run_noiseless_vs_noise_velocity_profile_comparison,
    run_velocity_profile_comparison_batch,
)


def main() -> None:
    algorithm_name = "abqlbm"  # Options: "abqlbm", "msqlbm", "spacetime"
    hardware_name = "superconducting_ibm_eagle_r3_2024"
    hardware_names = [
        "superconducting_ibm_nighthawk_r1_2025",
        "neutral_atom_aws_2023",
    ]
    velocity_profile_timesteps = [1, 3]
    max_timesteps = 5
    num_shots = 4096
    optimization_level = 1
    seed_transpiler = 42
    seed_simulator = 42
    backend_method = "statevector"

    # none, depolarizing, hardware_depolarizing, thermal_relaxation
    noise_kind = "hardware_depolarizing"
    noise_parameters = {
        "single_qubit_probability": 0.000001,
        "two_qubit_probability": 0.00001,
    }
    hardware_config_overrides = {
        "neutral_atom_aws_2023": {
            "coherence": {
                "t1_s": 4.0,
                "t2_s": 1.0,
            }
        }
    }
    # Options:
    # "hardware_run"
    # "density_comparison"
    # "density_error_growth"
    # "velocity_profile_comparison"
    # "velocity_profile_batch"
    # "multi_hardware_velocity"
    # "multi_hardware_density"
    # "depolarizing_sweep"
    experiment = "velocity_profile_batch"

    if experiment == "velocity_profile_batch":
        output_dir = run_velocity_profile_comparison_batch(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            final_timestep=1,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    elif experiment == "multi_hardware_density":
        output_dir = run_multi_hardware_density_depolarizing_comparison(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            backend_method=backend_method,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    elif experiment == "multi_hardware_velocity":
        output_dir = run_multi_hardware_velocity_profile_comparison(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            timesteps=velocity_profile_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            hardware_config_overrides=hardware_config_overrides,
        )
    elif experiment == "depolarizing_sweep":
        output_dir = run_depolarizing_probability_sweep(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            timestep=1,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            two_qubit_probabilities=[
                0.0,
                1e-5,
                3e-5,
                1e-4,
                3e-4,
                1e-3,
                3e-3,
                1e-2,
            ],
            backend_method=backend_method,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            single_qubit_probability_scale=0.1,
        )
    elif experiment == "velocity_profile_comparison":
        output_dir = run_noiseless_vs_noise_velocity_profile_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            final_timestep=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    elif experiment == "density_error_growth":
        output_dir = run_density_error_growth_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    elif experiment == "density_comparison":
        output_dir = run_noiseless_vs_noise_density_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            final_timestep=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    elif experiment == "hardware_run":
        output_dir = run_hardware_noise_analysis(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )
    else:
        raise ValueError(f"Unknown experiment {experiment!r}.")

    print(f"QLBM hardware noise-analysis output: {output_dir}")
    print(f"Analysis files: {output_dir / 'analysis'}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
