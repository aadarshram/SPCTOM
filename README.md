# SPCTOM: Smooth Path-Constrained Time-Optimal Motion Planning
---

## Table of contents

- [SPCTOM: Smooth Path-Constrained Time-Optimal Motion Planning](#spctom-smooth-path-constrained-time-optimal-motion-planning)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [TODO](#todo)
  - [Repo structure](#repo-structure)
  - [Instructions for use](#instructions-for-use)
  - [How it works](#how-it-works)
  - [References](#references)

---

## Introduction

This repository implements a path-constrained time optimal trajectory planner with both torque and torque-rate constraints. It's a inspired reimplementation of the methods described in [1], [2], [3].   
*This project is a part of my course EE6412: Optimal Control at IIT Madras.*
- **PCTOM**: Solves the time optimal control problem in the phase-plane under torque constraints. The solution follows the classic bang-singular-bang control structure.
- **SPCTOM**: Builds on top of PCTOM handling torque rate constrains as well. Formulates the phase-plane solution and uses a Flexible Tolerance Polyhedron (FTP) style optimization to solve for a smooth, near time-optimal solution.

---

## TODO

- [x] Implement base structure
- [x] Test 1 DoF case
- [ ] Make robust switching logic in PCTOM
- [ ] Improve SPCTOM optimizer

---

## Repo structure

```text
SPCTOM/
├── src/
│   ├── main.py        # End-to-end run: setup, solve, compare, plot
│   ├── config.py      # Robot selection, limits, boundary conditions
│   ├── robot.py       # Robot dynamics/kinematics models
│   ├── utils.py       # Path parametrization utilities
│   ├── tom.py         # Shared time-optimal-motion base functionality
│   ├── pctom.py       # Phase-plane constrained time-optimal solver
│   └── spctom.py      # Smooth constrained optimization-based solver
├── docs/              # Project docs/notes
├── pctom_paper.pdf
├── spctom_thesis.pdf
├── pyproject.toml
└── README.md
```

---

## Instructions for use

1. Create/activate environment and install dependencies.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Select robot and limits in `src/config.py`:
	- `ROBOT = 'CAR_1DOF'` for simple benchmark
	- `ROBOT = 'ELBOW_3DOF'` for manipulator case
3. Run:

```bash
python src/main.py
```

The script will:

- solve with PCTOM,
- warm-start and solve with SPCTOM,
- print timing/jerk comparison,
- open dashboard plots (PCTOM, SPCTOM, and overlay comparison).

---

## How it works

A detailed explanation of the methods and implementation can be found in the [presentation](LINK TO BE ADDED) and [report](LINK TO BE ADDED).

---

## References

1. D. Constantinescu and E. A. Croft, *Smooth and Time-Optimal Trajectory Planning for Industrial Manipulators along Specified Paths*, Journal of Robotic Systems, 2000.
2. J. E. Bobrow, S. Dubowsky, and J. S. Gibson, *Time-optimal control of robotic manipulators along specified paths*, The International Journal of Robotics Research, 1985.
3. A. Shiller and H.-H. Lu, *Computation of Path Constrained Time Optimal Motions With Dynamic Singularities*.

---