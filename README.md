# qlbm-hardware

Support code and result artifacts for the MSc thesis *Towards Hardware
Implementation of Quantum Lattice Boltzmann Methods*.

This repository contains the thesis code for evaluating
[`qlbm`](https://github.com/QCFD-Lab/qlbm) generated QLBM circuits under
hardware constraints: resource estimation and transpilation, phase-polynomial
optimization, and noisy simulation of density and velocity observables. It also
keeps the hardware-model data and selected result artifacts used in the thesis.

## Layout

- `components/resource_estimator/`: custom `qlbm` resource-estimation framework.
- `components/phase_poly_optimizer/`: phase-polynomial optimization for `qlbm` circuits.
- `components/noise_analysis/`: noise analysis for `qlbm` circuits.

The resource-estimation framework is used by both the phase-polynomial optimization and noise-analysis workflows.

## Running

The scripts assume that `qlbm` and its dependencies are installed, including
Qiskit, pytket, and pandas. In the thesis workspace this was done with a
shared `qlbm-cpu-venv` environment.

---

This is research code accompanying the thesis, not a packaged library.
