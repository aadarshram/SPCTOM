"""
Diagnostic script to understand SPCTOM warm-start degradation.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.robot import Robot
from src.utils import PathParametrize
from src.spctom import SPCTOM, build_knots_from_pctom
from src.pctom import PCTOM
from src.config import (
    ROBOT,
    T_MAX,
    T_MIN,
    T_DOT_MAX,
    T_DOT_MIN,
    q0, qf, q0_dot, qf_dot, q0_ddot, qf_ddot,
)


def check_feasibility(planner, s_check, sdot, sddot, sdddot, phi=1e-6):
    """
    Check torque and torque-rate constraints.
    Returns:
      - T_violation: max normalized torque violation (0 = feasible)
      - Tdot_violation: max normalized torque-rate violation
    """
    a_T = np.zeros((len(s_check), planner.robot.n_joints()))
    b_T = np.zeros_like(a_T)
    c_T = np.zeros_like(a_T)
    a_Tdot = np.zeros_like(a_T)
    b_Tdot = np.zeros_like(a_T)
    c_Tdot = np.zeros_like(a_T)
    d_Tdot = np.zeros_like(a_T)

    for i, s in enumerate(s_check):
        a_T[i], b_T[i], c_T[i] = planner.get_T_coeffs(float(s))
        a_Tdot[i], b_Tdot[i], c_Tdot[i], d_Tdot[i] = planner.get_T_dot_coeffs(float(s))

    tau = a_T * sddot[:, None] + b_T * (sdot[:, None]**2) + c_T
    tau_dot = (a_Tdot * sdddot[:, None] + 
               b_Tdot * (sdot[:, None] * sddot[:, None]) +
               c_Tdot * (sdot[:, None]**3) +
               d_Tdot * sdot[:, None])

    # Torque violations
    T_vio = 0.0
    for j in range(planner.robot.n_joints()):
        if planner.T_max[j] < np.inf:
            v = tau[:, j] / planner.T_max[j] - 1.0
            T_vio = max(T_vio, np.max(v))
        if planner.T_min[j] > -np.inf:
            v = tau[:, j] / planner.T_min[j] - 1.0
            T_vio = max(T_vio, np.max(v))

    # Torque-rate violations
    Tdot_vio = 0.0
    for j in range(planner.robot.n_joints()):
        if planner.T_dot_max[j] < np.inf:
            v = tau_dot[:, j] / planner.T_dot_max[j] - 1.0
            Tdot_vio = max(Tdot_vio, np.max(v))
        if planner.T_dot_min[j] > -np.inf:
            v = tau_dot[:, j] / planner.T_dot_min[j] - 1.0
            Tdot_vio = max(Tdot_vio, np.max(v))

    return T_vio, Tdot_vio


def main():
    # Setup robot
    robot = Robot(robot_type=ROBOT)
    print(f'Robot {robot.__str__()} with {robot.n_joints()} joints initialized.')

    # Setup path
    if ROBOT == 'CAR_1DOF':
        q0_local = q0.copy()
        qf_local = qf.copy()
        r0 = robot.forward_kinematics(q0_local)
        r1 = robot.forward_kinematics(qf_local)
        s_values = np.linspace(0, 1, num=2001)
    else:
        q_min, q_max = np.array([-np.pi, -np.pi, -np.pi]), np.array([np.pi, np.pi, np.pi])
        margin = 0.15
        q_min_safe = q_min + margin * (q_max - q_min)
        q_max_safe = q_max - margin * (q_max - q_min)
        q0_local = np.random.uniform(q_min_safe, q_max_safe)
        qf_local = np.random.uniform(q_min_safe, q_max_safe)
        r0 = robot.forward_kinematics(q0_local)
        r1 = robot.forward_kinematics(qf_local)
        s_values = np.linspace(0, 1, num=100)

    R_S = PathParametrize(r0, r1, method='linear')

    # Boundary conditions
    boundary_conditions = {
        'q': (q0_local, qf_local),
        'q_dot': (q0_dot, qf_dot),
        'q_ddot': (q0_ddot, qf_ddot),
    }
    ds = float(s_values[1] - s_values[0])

    # === PCTOM SOLVE ===
    print("\n[1] PCTOM solving...")
    planner = PCTOM(robot=robot, path_param=R_S, s_values=s_values, T_max=T_MAX, T_min=T_MIN, ds=ds, boundary_conditions=boundary_conditions)
    solution_pct = planner.solve(verbose=False)

    s_pct = np.asarray(solution_pct['s_traj'], dtype=float)
    sdot_pct = np.asarray(solution_pct['sdot_traj'], dtype=float)
    sddot_pct = np.asarray(solution_pct['sddot_traj'], dtype=float)
    motion_time_pct = solution_pct['motion_time']

    # Compute third derivative
    order = np.argsort(s_pct)
    s_pct = s_pct[order]
    sdot_pct = sdot_pct[order]
    sddot_pct = sddot_pct[order]
    _, idx = np.unique(np.round(s_pct, 12), return_index=True)
    s_pct = s_pct[idx]
    sdot_pct = sdot_pct[idx]
    sddot_pct = sddot_pct[idx]
    sdddot_pct = np.gradient(sddot_pct, s_pct, edge_order=1) * sdot_pct

    max_sdot_pct = float(np.max(sdot_pct))
    print(f"  PCTOM result: motion_time = {motion_time_pct:.6f} s, max_sdot = {max_sdot_pct:.6f}")

    # Check PCTOM feasibility
    T_vio_pct, Tdot_vio_pct = check_feasibility(planner, s_pct, sdot_pct, sddot_pct, sdddot_pct)
    print(f"  PCTOM feasibility: T_violation = {T_vio_pct:.2e}, Tdot_violation = {Tdot_vio_pct:.2e}")

    # === SPCTOM WARM-START SETUP ===
    print("\n[2] SPCTOM warm-start setup...")
    planner_sp = SPCTOM(
        robot=robot, path_param=R_S, s_values=s_values,
        T_max=T_MAX, T_min=T_MIN,
        T_dot_max=T_DOT_MAX, T_dot_min=T_DOT_MIN,
        boundary_conditions=boundary_conditions, ds=ds,
    )

    planner_sp.warm_start_from_pctom(
        solution_pct,
        n_uniform_fill=4,
        boundary_pad=1,
        min_knots=16,
    )

    print(f"  Warm-start knots: {len(planner_sp.s_knots)} knots")
    print(f"  Warm-start v_init: [{planner_sp.v_init[0]:.4f}, ..., {planner_sp.v_init[-1]:.4f}]")
    print(f"  Warm-start boundary: ({planner_sp.v_init[0]:.4f}, {planner_sp.v_init[-1]:.4f})")

    # Check warm-start spline feasibility at knot locations
    s_knots = planner_sp.s_knots
    v_init = planner_sp.v_init
    sp_init = planner_sp._build_spline(v_init)

    s_eval = np.linspace(s_knots[0], s_knots[-1], 200)
    sdot_init = np.maximum(sp_init(s_eval), 1e-9)
    sdot_deriv_init = sp_init(s_eval, 1)
    sdot_deriv2_init = sp_init(s_eval, 2)
    sddot_init = sdot_deriv_init * sdot_init
    sdddot_init = sdot_init * (sdot_deriv_init**2 + sdot_init * sdot_deriv2_init)

    T_vio_init, Tdot_vio_init = check_feasibility(planner_sp, s_eval, sdot_init, sddot_init, sdddot_init)
    print(f"  Warm-start spline feasibility: T_violation = {T_vio_init:.2e}, Tdot_violation = {Tdot_vio_init:.2e}")

    # === INSPECT INITIAL POLYHEDRON ===
    print("\n[3] Inspecting initial polyhedron feasibility...")
    n = len(planner_sp.s_knots)
    bc = (planner_sp.v_init[0], planner_sp.v_init[-1])

    def _apply_bc(x):
        x = x.copy()
        x[0] = bc[0]
        x[-1] = bc[1]
        return np.maximum(x, 1e-9)

    def objective(x):
        sp = planner_sp._build_spline(_apply_bc(x))
        sdot = np.maximum(sp(np.linspace(s_knots[0], s_knots[-1], 400)), 1e-9)
        trapz_fn = getattr(np, 'trapezoid', np.trapz)
        return float(trapz_fn(1.0 / sdot, np.linspace(s_knots[0], s_knots[-1], 400)))

    def constraint_violation(x):
        sp = planner_sp._build_spline(_apply_bc(x))
        s_check = np.linspace(s_knots[0], s_knots[-1], 100)
        sdot = np.maximum(sp(s_check), 1e-9)
        sdot_deriv = sp(s_check, 1)
        sdot_deriv2 = sp(s_check, 2)
        sddot = sdot_deriv * sdot
        sdddot = sdot * (sdot_deriv**2 + sdot * sdot_deriv2)
        T_vio, Tdot_vio = check_feasibility(planner_sp, s_check, sdot, sddot, sdddot)
        return max(T_vio, Tdot_vio)

    # Check initial guess
    x0 = _apply_bc(np.array(planner_sp.v_init, dtype=float))
    f0 = objective(x0)
    T0 = constraint_violation(x0)
    print(f"  Initial guess (v_init): f = {f0:.6f}, constraint_vio = {T0:.2e}")

    # Check a uniform slow guess
    x_slow = _apply_bc(np.full(n, 1e-4))
    f_slow = objective(x_slow)
    T_slow = constraint_violation(x_slow)
    print(f"  Slow guess (1e-4): f = {f_slow:.6f}, constraint_vio = {T_slow:.2e}")

    # === SPCTOM SOLVE ===
    print("\n[4] SPCTOM solving...")
    solution_sp = planner_sp.solve(
        max_iter=500,
        phi0=0.05,
        k_phi=0.25,
        tol=1e-6,
        verbose=True,
    )

    motion_time_sp = solution_sp['motion_time']
    s_sp_raw = np.asarray(s_values, dtype=float)
    sdot_sp_raw = np.maximum(solution_sp['spline'](s_sp_raw), 1e-9)
    max_sdot_sp = float(np.max(sdot_sp_raw))

    print(f"  SPCTOM result: motion_time = {motion_time_sp:.6f} s, max_sdot = {max_sdot_sp:.6f}")
    print(f"  Iterations: {solution_sp['iterations']}, phi_final: {solution_sp['phi_final']:.2e}")

    # === SUMMARY ===
    print("\n[SUMMARY]")
    print(f"  PCTOM  : time = {motion_time_pct:.6f} s, max_sdot = {max_sdot_pct:.6f}")
    print(f"  SPCTOM : time = {motion_time_sp:.6f} s, max_sdot = {max_sdot_sp:.6f}")
    print(f"  Degradation: {motion_time_sp / motion_time_pct:.2f}x slower")
    print(f"  Speed loss: {(1 - max_sdot_sp / max_sdot_pct) * 100:.1f}%")

    # === PLOTTING ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: PCTOM vs SPCTOM profiles
    ax = axes[0, 0]
    ax.plot(s_pct, sdot_pct, 'b-', linewidth=2, label='PCTOM sdot')
    ax.plot(s_sp_raw, sdot_sp_raw, 'r--', linewidth=2, label='SPCTOM sdot')
    ax.set_xlabel('s')
    ax.set_ylabel('sdot')
    ax.set_title('Speed Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: PCTOM vs SPCTOM knots
    ax = axes[0, 1]
    ax.scatter(s_pct, sdot_pct, alpha=0.5, s=20, label='PCTOM trajectory')
    ax.scatter(planner_sp.s_knots, planner_sp.v_init, color='red', s=100, marker='x', label='SPCTOM warm-start knots')
    ax.scatter(planner_sp.s_knots, solution_sp['sdot_knots'], color='orange', s=100, marker='^', label='SPCTOM optimized knots')
    ax.set_xlabel('s')
    ax.set_ylabel('sdot')
    ax.set_title('Knot Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Acceleration comparison
    ax = axes[1, 0]
    ax.plot(s_pct, sddot_pct, 'b-', linewidth=2, label='PCTOM sddot')
    sddot_sp = solution_sp['spline'](s_sp_raw, 1) * sdot_sp_raw
    ax.plot(s_sp_raw, sddot_sp, 'r--', linewidth=2, label='SPCTOM sddot')
    ax.set_xlabel('s')
    ax.set_ylabel('sddot')
    ax.set_title('Accelerations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence info (text box)
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    PCTOM Results:
      motion_time = {motion_time_pct:.6f} s
      max_sdot = {max_sdot_pct:.6f}
      T_violation = {T_vio_pct:.2e}
      Tdot_violation = {Tdot_vio_pct:.2e}

    Warm-start Spline:
      T_violation = {T_vio_init:.2e}
      Tdot_violation = {Tdot_vio_init:.2e}

    SPCTOM Results:
      motion_time = {motion_time_sp:.6f} s
      max_sdot = {max_sdot_sp:.6f}
      iterations = {solution_sp['iterations']}
      phi_final = {solution_sp['phi_final']:.2e}

    Initial Feasibility:
      v_init: f={f0:.4e}, vio={T0:.2e}
      slow:   f={f_slow:.4e}, vio={T_slow:.2e}
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig('/tmp/diagnostic.png', dpi=100)
    print("\nPlot saved to /tmp/diagnostic.png")


if __name__ == '__main__':
    main()
