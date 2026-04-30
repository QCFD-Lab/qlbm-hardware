"""Configuration helpers for tube-flow noise-analysis experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "tube_analysis"


@dataclass(frozen=True)
class TubeConfig:
    """Geometry and discretization settings for a 2D D2Q9 tube."""

    width: int = 16
    height: int = 8
    velocities: str = "D2Q9"

    def validate(self) -> None:
        if self.width < 4 or self.height < 4:
            raise ValueError("Tube dimensions must be at least 4x4.")
        if self.width & (self.width - 1) or self.height & (self.height - 1):
            raise ValueError("QLBM AB lattices require power-of-two dimensions.")
        if self.velocities.upper() != "D2Q9":
            raise ValueError("The tube analysis currently supports D2Q9 only.")


@dataclass(frozen=True)
class BeamConfig:
    """Single-particle inlet beam configuration."""

    x: int = 0
    y: int | None = None
    velocity_channel: int = 5

    def resolved_y(self, tube: TubeConfig) -> int:
        return tube.height // 2 if self.y is None else self.y

    def validate(self, tube: TubeConfig) -> None:
        y = self.resolved_y(tube)
        if not 0 <= self.x < tube.width:
            raise ValueError(f"Beam x={self.x} is outside [0, {tube.width - 1}].")
        if not 0 <= y < tube.height:
            raise ValueError(f"Beam y={y} is outside [0, {tube.height - 1}].")
        if y in (0, tube.height - 1):
            raise ValueError("The beam should start on an interior row, not a wall.")
        if not 0 <= self.velocity_channel <= 8:
            raise ValueError("D2Q9 velocity channel must be in [0, 8].")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise-model selection and parameters."""

    kind: str = "none"
    single_qubit_probability: float = 0.001
    two_qubit_probability: float = 0.01
    damping_probability: float = 0.001
    phase_probability: float = 0.001
    gate_error_probability: float = 0.001

    def validate(self) -> None:
        allowed = {"none", "depolarizing", "amplitude_damping", "phase_damping", "gate_error"}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported noise kind {self.kind!r}; expected one of {sorted(allowed)}.")
        for name, value in asdict(self).items():
            if name == "kind":
                continue
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be a probability in [0, 1].")


@dataclass(frozen=True)
class RunConfig:
    """Execution settings for a tube-analysis run."""

    num_steps: int = 1
    num_shots: int = 4096
    optimization_level: int = 0
    output_root: Path = DEFAULT_OUTPUT_ROOT
    backend_method: str | None = None
    create_plots: bool = True
    config_only: bool = False

    def validate(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if self.num_shots <= 0:
            raise ValueError("num_shots must be positive.")
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2, or 3.")


@dataclass(frozen=True)
class TubeAnalysisConfig:
    """Full configuration for a tube-flow noise-analysis run."""

    tube: TubeConfig = field(default_factory=TubeConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> None:
        self.tube.validate()
        self.beam.validate(self.tube)
        self.noise.validate()
        self.run.validate()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run"]["output_root"] = str(self.run.output_root)
        return data

    def run_directory(self) -> Path:
        beam_y = self.beam.resolved_y(self.tube)
        name = (
            f"{self.noise.kind}_"
            f"{self.tube.width}x{self.tube.height}_"
            f"v{self.beam.velocity_channel}_"
            f"x{self.beam.x}_y{beam_y}_"
            f"steps{self.run.num_steps}_shots{self.run.num_shots}"
        )
        return self.run.output_root / name


def build_tube_lattice_config(tube: TubeConfig) -> dict[str, Any]:
    """Return an ABLattice-compatible tube configuration."""

    tube.validate()
    return {
        "lattice": {
            "dim": {"x": tube.width, "y": tube.height},
            "velocities": tube.velocities,
        },
        "geometry": [
            {
                "shape": "cuboid",
                "x": [0, tube.width - 1],
                "y": [0, 0],
                "boundary": "bounceback",
            },
            {
                "shape": "cuboid",
                "x": [0, tube.width - 1],
                "y": [tube.height - 1, tube.height - 1],
                "boundary": "bounceback",
            },
        ],
    }

