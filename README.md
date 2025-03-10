# Numerical stabilization of the Boussinesq system using boundary feedback control
This repository provides the source codes for the experiments in our paper for Feedback stabilization of the Boussinesq system.
If you use this code, please cite
```
@article{chandrashekar2021numerical,
  title={Numerical stabilization of the Boussinesq system using boundary feedback control},
  author={Chandrashekar, Praveen and Ramaswamy, Mythily and Raymond, Jean-Pierre and Sandilya, Ruchi},
  journal={Computers \& Mathematics with Applications},
  volume={89},
  pages={163--183},
  year={2021},
  publisher={Elsevier}
}
```

Python implementations for solving the Navier-Stokes-Boussinesq equations with boundary feedback controls. The code provides tools for obtaining stationary solutions, solving the linearized system, and running a nonlinear dynamical system.

## Usage

### 1. Run `Steady.py`
Use `Steady.py` to compute stationary solutions.

```bash
python Steady.py
```

### 2. Run `linear.py`
Use `linear.py` to solve the linearized system.

```bash
python linear.py
```

### 3. Run `gain.m`
Use `gain.m` to compute the gain matrix.

```bash
matlab gain.m
```

### 4. Run `run.py`
Use `run.py` to solve the full nonlinear dynamical system.

```bash
python run.py
```
