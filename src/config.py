# Imports
import numpy as np
from pathlib import Path

ROBOT = 'CAR_1DOF' # 'ELBOW_3DOF' or 'CAR_1DOF'

# Robot constraints + boundary conditions per robot
if ROBOT == 'CAR_1DOF':
	# Canonical double-integrator / 1D car benchmark
	T_MAX = np.array([1.0], dtype=float)
	T_MIN = -T_MAX
	T_DOT_MAX = np.array([10], dtype=float)
	T_DOT_MIN = -T_DOT_MAX

	q0 = np.array([0.0], dtype=float)
	qf = np.array([1.0], dtype=float)
	q0_dot = np.array([1e-3], dtype=float)
	qf_dot = np.array([1e-3], dtype=float)
	q0_ddot = np.array([0.0], dtype=float)
	qf_ddot = np.array([0.0], dtype=float)
else:
	# 3-DOF elbow manipulator defaults
	T_MAX = np.array([75, 75, 75], dtype=float)
	T_MIN = -T_MAX
	cat = {'High': 3, 'Medium': 2, 'Low': 1}
	T_DOT_MAX = np.ones_like(T_MAX) * (10 ** cat['High'])
	T_DOT_MIN = -T_DOT_MAX

	q0 = np.array([0.0, 0.0, 0.0])
	qf = np.array([1.0, 1.0, -1.0])
	q0_dot = np.zeros_like(q0)
	qf_dot = np.zeros_like(qf)
	q0_ddot = np.zeros_like(q0)
	qf_ddot = np.zeros_like(qf)

# Objective
J = 'min_time'
# Problem constraints
tf = 5.0 # Max time for the trajectory (seconds)