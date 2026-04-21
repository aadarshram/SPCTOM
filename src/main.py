import numpy as np
import matplotlib.pyplot as plt

from src.robot import Robot
from src.utils import PathParametrize
from src.spctom import SPCTOM
from src.pctom import PCTOM
from src.config import (
    ROBOT,
    T_MAX,
    T_MIN,
    T_DOT_MAX,
    T_DOT_MIN,
    q0, qf, q0_dot, qf_dot, q0_ddot, qf_ddot,
) 

def main():

    #  Setup robot

    robot = Robot(
        robot_type=ROBOT,
    )
    print(f'Robot {robot.__str__()} with {robot.n_joints()} joints initialized.')
    # Setup path
    if ROBOT == 'CAR_1DOF':
        # Known benchmark: 1D path from 0 -> 1
        q0_local = q0.copy()
        qf_local = qf.copy()
        r0 = robot.forward_kinematics(q0_local)
        r1 = robot.forward_kinematics(qf_local)
        s_values = np.linspace(0, 1, num=2001)
    else:
        # Randomized task-space segment for elbow robot
        q_min, q_max = np.array([-np.pi, -np.pi, -np.pi]), np.array([np.pi, np.pi, np.pi])
        margin = 0.15
        q_min_safe = q_min + margin * (q_max - q_min)
        q_max_safe = q_max - margin * (q_max - q_min)
        q0_local = np.random.uniform(q_min_safe, q_max_safe)
        qf_local = np.random.uniform(q_min_safe, q_max_safe)
        r0 = robot.forward_kinematics(q0_local)
        r1 = robot.forward_kinematics(qf_local)
        s_values = np.linspace(0, 1, num=100)

    print(f"Start task-space position: {r0}")
    print(f"End task-space position: {r1}")
    R_S = PathParametrize(r0, r1, method='linear')

    # Setup planner

    # Boundary conditions
    boundary_conditions = {
        'q': (q0_local, qf_local),
        'q_dot': (q0_dot, qf_dot),
        'q_ddot': (q0_ddot, qf_ddot),
    }
    ds = float(s_values[1] - s_values[0])
    planner = PCTOM(robot=robot, path_param=R_S, s_values=s_values, T_max=T_MAX, T_min=T_MIN, ds=ds, boundary_conditions=boundary_conditions)
    # Solve
    solution = planner.solve(verbose=True)
    print(solution)

    if ROBOT == 'CAR_1DOF':
        s = solution['s_traj']
        sdot = solution['sdot_traj']
        s_peak = float(s[np.argmax(sdot)])
        print(f"[CAR_1DOF benchmark] peak s ≈ {s_peak:.4f} (expected ≈ 0.5)")
        print(f"[CAR_1DOF benchmark] max sdot ≈ {float(np.max(sdot)):.4f} (expected ≈ 1.0)")
        print(f"[CAR_1DOF benchmark] motion time ≈ {float(solution['motion_time']):.4f} (expected ≈ 2.0)")

    def _prepare_profile(s, sdot, sddot=None):
        s = np.asarray(s, dtype=float)
        sdot = np.maximum(np.asarray(sdot, dtype=float), 1e-9)
        order = np.argsort(s)
        s = s[order]
        sdot = sdot[order]
        if sddot is not None:
            sddot = np.asarray(sddot, dtype=float)[order]
        # Deduplicate nearly-identical s values for stable derivatives.
        _, idx = np.unique(np.round(s, 12), return_index=True)
        s = s[idx]
        sdot = sdot[idx]
        if sddot is not None:
            sddot = sddot[idx]
        if sddot is None:
            v = sdot**2
            sddot = 0.5 * np.gradient(v, s, edge_order=1)
        sdddot = np.gradient(sddot, s, edge_order=1) * sdot
        return s, sdot, sddot, sdddot

    def _compute_tau_and_sddot_bounds(planner_like, s, sdot, sddot):
        tau = np.zeros((len(s), robot.n_joints()), dtype=float)
        sddot_min = np.zeros(len(s), dtype=float)
        sddot_max = np.zeros(len(s), dtype=float)
        for i, (si, sdi, sddi) in enumerate(zip(s, sdot, sddot)):
            a_s, b_s, c_s = planner_like.get_T_coeffs(float(si))
            tau[i, :] = a_s * sddi + b_s * (sdi**2) + c_s
            lo, hi = planner_like._sddot_bounds_from_T(
                a_s, b_s, c_s,
                tau_min=T_MIN,
                tau_max=T_MAX,
                sdot=float(max(sdi, 1e-9)),
            )
            sddot_min[i] = lo
            sddot_max[i] = hi
        return tau, sddot_min, sddot_max

    def _compute_tau_dot(planner_like, s, sdot, sddot, sdddot):
        tau_dot = np.zeros((len(s), robot.n_joints()), dtype=float)
        for i, (si, sdi, sddi, sdddi) in enumerate(zip(s, sdot, sddot, sdddot)):
            a_d, b_d, c_d, d_d = planner_like.get_T_dot_coeffs(float(si))
            tau_dot[i, :] = (
                a_d * sdddi
                + b_d * sdi * sddi
                + c_d * (sdi**3)
                + d_d * sdi
            )
        return tau_dot

    # Build PCTOM profile quantities
    s_pct, sdot_pct, sddot_pct, sdddot_pct = _prepare_profile(
        solution['s_traj'], solution['sdot_traj'], solution['sddot_traj']
    )
    tau_pct, sddot_min_pct, sddot_max_pct = _compute_tau_and_sddot_bounds(
        planner, s_pct, sdot_pct, sddot_pct
    )
    vlc_pct = np.asarray([planner._sdot_max_at(float(si)) for si in s_pct], dtype=float)

    # Try with SPCTOM
    planner_spctom = SPCTOM(
        robot=robot,
        path_param=R_S,
        s_values=s_values,
        T_max=T_MAX,
        T_min=T_MIN,
        T_dot_max=T_DOT_MAX,
        T_dot_min=T_DOT_MIN,
        boundary_conditions=boundary_conditions,
        ds=ds,
    )
    planner_spctom.warm_start_from_pctom(
        solution,
        n_uniform_fill=4,
        boundary_pad=1,
        min_knots=16,
    )
    solution_spctom = planner_spctom.solve(
        max_iter=500,
        phi0=0.05,
        k_phi=0.25,
        tol=1e-6,
        verbose=True,
    )
    print(solution_spctom)

    # Build SPCTOM profile quantities
    s_sp_raw = np.asarray(s_values, dtype=float)
    sdot_sp_raw = np.maximum(solution_spctom['spline'](s_sp_raw), 1e-9)
    s_sp, sdot_sp, sddot_sp, sdddot_sp = _prepare_profile(s_sp_raw, sdot_sp_raw)
    tau_sp, sddot_min_sp, sddot_max_sp = _compute_tau_and_sddot_bounds(
        planner_spctom, s_sp, sdot_sp, sddot_sp
    )
    tau_dot_sp = _compute_tau_dot(planner_spctom, s_sp, sdot_sp, sddot_sp, sdddot_sp)
    vlc_sp = np.asarray([planner._sdot_max_at(float(si)) for si in s_sp], dtype=float)

    tau_dot_pct = _compute_tau_dot(planner_spctom, s_pct, sdot_pct, sddot_pct, sdddot_pct)

    jerk_rms_pct = float(np.sqrt(np.mean(sdddot_pct**2)))
    jerk_rms_sp = float(np.sqrt(np.mean(sdddot_sp**2)))
    print(f"[Compare] PCTOM time = {float(solution['motion_time']):.6f}, jerk_rms = {jerk_rms_pct:.6f}")
    print(f"[Compare] SPCTOM time = {float(solution_spctom['motion_time']):.6f}, jerk_rms = {jerk_rms_sp:.6f}")

    # ================================================================
    # Figure 1: PCTOM dashboard
    # ================================================================
    fig_pct, ax_pct = plt.subplots(2, 2, figsize=(14, 9), num='PCTOM')
    fig_pct.suptitle(f"PCTOM | motion time = {float(solution['motion_time']):.6f} s", fontsize=13)

    ax_pct[0, 0].plot(s_pct, sdot_pct, linewidth=2.0, label='sdot')
    ax_pct[0, 0].plot(s_pct, vlc_pct, linestyle='--', linewidth=1.5, label='sdot upper bound (VLC)')
    ax_pct[0, 0].set_title('sdot vs s')
    ax_pct[0, 0].set_xlabel('s')
    ax_pct[0, 0].set_ylabel('sdot')
    ax_pct[0, 0].grid(True, alpha=0.3)
    ax_pct[0, 0].legend()

    ax_pct[0, 1].plot(s_pct, sddot_pct, linewidth=2.0, label='sddot')
    ax_pct[0, 1].plot(s_pct, sddot_min_pct, linestyle='--', linewidth=1.4, label='sddot min bound')
    ax_pct[0, 1].plot(s_pct, sddot_max_pct, linestyle='--', linewidth=1.4, label='sddot max bound')
    ax_pct[0, 1].set_title('sddot vs s')
    ax_pct[0, 1].set_xlabel('s')
    ax_pct[0, 1].set_ylabel('sddot')
    ax_pct[0, 1].grid(True, alpha=0.3)
    ax_pct[0, 1].legend()

    for j in range(robot.n_joints()):
        ax_pct[1, 0].plot(s_pct, tau_pct[:, j], linewidth=1.8, label=f'tau_{j+1}')
        ax_pct[1, 0].axhline(T_MAX[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax_pct[1, 0].axhline(T_MIN[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_pct[1, 0].set_title('Torque vs s (with bounds)')
    ax_pct[1, 0].set_xlabel('s')
    ax_pct[1, 0].set_ylabel('tau')
    ax_pct[1, 0].grid(True, alpha=0.3)
    ax_pct[1, 0].legend()

    ax_pct[1, 1].plot(s_pct, sdddot_pct, linewidth=2.0, label='jerk (sdddot)')
    ax_pct[1, 1].set_title('Jerk vs s')
    ax_pct[1, 1].set_xlabel('s')
    ax_pct[1, 1].set_ylabel('sdddot')
    ax_pct[1, 1].grid(True, alpha=0.3)
    ax_pct[1, 1].legend()

    fig_pct.tight_layout(rect=(0, 0.02, 1, 0.96))

    # ================================================================
    # Figure 2: SPCTOM dashboard
    # ================================================================
    fig_sp, ax_sp = plt.subplots(3, 2, figsize=(14, 12), num='SPCTOM')
    fig_sp.suptitle(f"SPCTOM | motion time = {float(solution_spctom['motion_time']):.6f} s", fontsize=13)

    ax_sp[0, 0].plot(s_sp, sdot_sp, linewidth=2.0, label='sdot')
    ax_sp[0, 0].plot(s_sp, vlc_sp, linestyle='--', linewidth=1.5, label='sdot upper bound (VLC)')
    ax_sp[0, 0].set_title('sdot vs s')
    ax_sp[0, 0].set_xlabel('s')
    ax_sp[0, 0].set_ylabel('sdot')
    ax_sp[0, 0].grid(True, alpha=0.3)
    ax_sp[0, 0].legend()

    ax_sp[0, 1].plot(s_sp, sddot_sp, linewidth=2.0, label='sddot')
    ax_sp[0, 1].plot(s_sp, sddot_min_sp, linestyle='--', linewidth=1.4, label='sddot min bound')
    ax_sp[0, 1].plot(s_sp, sddot_max_sp, linestyle='--', linewidth=1.4, label='sddot max bound')
    ax_sp[0, 1].set_title('sddot vs s')
    ax_sp[0, 1].set_xlabel('s')
    ax_sp[0, 1].set_ylabel('sddot')
    ax_sp[0, 1].grid(True, alpha=0.3)
    ax_sp[0, 1].legend()

    for j in range(robot.n_joints()):
        ax_sp[1, 0].plot(s_sp, tau_sp[:, j], linewidth=1.8, label=f'tau_{j+1}')
        ax_sp[1, 0].axhline(T_MAX[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax_sp[1, 0].axhline(T_MIN[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_sp[1, 0].set_title('Torque vs s (with bounds)')
    ax_sp[1, 0].set_xlabel('s')
    ax_sp[1, 0].set_ylabel('tau')
    ax_sp[1, 0].grid(True, alpha=0.3)
    ax_sp[1, 0].legend()

    ax_sp[1, 1].plot(s_sp, sdddot_sp, linewidth=2.0, label='sdddot')
    ax_sp[1, 1].set_title('sdddot vs s')
    ax_sp[1, 1].set_xlabel('s')
    ax_sp[1, 1].set_ylabel('sdddot')
    ax_sp[1, 1].grid(True, alpha=0.3)
    ax_sp[1, 1].legend()

    for j in range(robot.n_joints()):
        ax_sp[2, 0].plot(s_sp, tau_dot_sp[:, j], linewidth=1.8, label=f'tau_dot_{j+1}')
        ax_sp[2, 0].axhline(T_DOT_MAX[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax_sp[2, 0].axhline(T_DOT_MIN[j], color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_sp[2, 0].set_title('Torque rate vs s (with bounds)')
    ax_sp[2, 0].set_xlabel('s')
    ax_sp[2, 0].set_ylabel('tau_dot')
    ax_sp[2, 0].grid(True, alpha=0.3)
    ax_sp[2, 0].legend()

    ax_sp[2, 1].axis('off')
    ax_sp[2, 1].text(
        0.02, 0.95,
        f"time = {float(solution_spctom['motion_time']):.6f} s\n"
        f"jerk_rms = {jerk_rms_sp:.6f}",
        va='top', ha='left', fontsize=11,
    )

    fig_sp.tight_layout(rect=(0, 0.02, 1, 0.96))

    # ================================================================
    # Figure 3: PCTOM vs SPCTOM overlay
    # ================================================================
    fig_cmp, ax_cmp = plt.subplots(3, 2, figsize=(14, 12), num='PCTOM vs SPCTOM')
    fig_cmp.suptitle(
        f"PCTOM vs SPCTOM | t_pct={float(solution['motion_time']):.6f}s, t_sp={float(solution_spctom['motion_time']):.6f}s",
        fontsize=13,
    )

    ax_cmp[0, 0].plot(s_pct, sdot_pct, linewidth=2.0, label='PCTOM sdot')
    ax_cmp[0, 0].plot(s_sp, sdot_sp, linewidth=2.0, linestyle='--', label='SPCTOM sdot')
    ax_cmp[0, 0].plot(s_pct, vlc_pct, linewidth=1.2, linestyle=':', label='VLC')
    ax_cmp[0, 0].set_title('sdot overlay')
    ax_cmp[0, 0].set_xlabel('s')
    ax_cmp[0, 0].set_ylabel('sdot')
    ax_cmp[0, 0].grid(True, alpha=0.3)
    ax_cmp[0, 0].legend()

    ax_cmp[0, 1].plot(s_pct, sddot_pct, linewidth=2.0, label='PCTOM sddot')
    ax_cmp[0, 1].plot(s_sp, sddot_sp, linewidth=2.0, linestyle='--', label='SPCTOM sddot')
    ax_cmp[0, 1].set_title('sddot overlay')
    ax_cmp[0, 1].set_xlabel('s')
    ax_cmp[0, 1].set_ylabel('sddot')
    ax_cmp[0, 1].grid(True, alpha=0.3)
    ax_cmp[0, 1].legend()

    for j in range(robot.n_joints()):
        ax_cmp[1, 0].plot(s_pct, tau_pct[:, j], linewidth=1.7, label=f'PCTOM tau_{j+1}')
        ax_cmp[1, 0].plot(s_sp, tau_sp[:, j], linewidth=1.7, linestyle='--', label=f'SPCTOM tau_{j+1}')
    ax_cmp[1, 0].set_title('Torque overlay')
    ax_cmp[1, 0].set_xlabel('s')
    ax_cmp[1, 0].set_ylabel('tau')
    ax_cmp[1, 0].grid(True, alpha=0.3)
    ax_cmp[1, 0].legend(ncol=2)

    ax_cmp[1, 1].plot(s_pct, sdddot_pct, linewidth=2.0, label='PCTOM jerk')
    ax_cmp[1, 1].plot(s_sp, sdddot_sp, linewidth=2.0, linestyle='--', label='SPCTOM sdddot')
    ax_cmp[1, 1].set_title('Jerk overlay')
    ax_cmp[1, 1].set_xlabel('s')
    ax_cmp[1, 1].set_ylabel('sdddot')
    ax_cmp[1, 1].grid(True, alpha=0.3)
    ax_cmp[1, 1].legend()

    for j in range(robot.n_joints()):
        ax_cmp[2, 0].plot(s_pct, tau_dot_pct[:, j], linewidth=1.7, label=f'PCTOM tau_dot_{j+1}')
        ax_cmp[2, 0].plot(s_sp, tau_dot_sp[:, j], linewidth=1.7, linestyle='--', label=f'SPCTOM tau_dot_{j+1}')
        ax_cmp[2, 0].axhline(T_DOT_MAX[j], color='k', linestyle=':', linewidth=0.8, alpha=0.45)
        ax_cmp[2, 0].axhline(T_DOT_MIN[j], color='k', linestyle=':', linewidth=0.8, alpha=0.45)
    ax_cmp[2, 0].set_title('Torque rate overlay (with bounds)')
    ax_cmp[2, 0].set_xlabel('s')
    ax_cmp[2, 0].set_ylabel('tau_dot')
    ax_cmp[2, 0].grid(True, alpha=0.3)
    ax_cmp[2, 0].legend(ncol=2)

    ax_cmp[2, 1].axis('off')
    ax_cmp[2, 1].text(
        0.02, 0.95,
        f"PCTOM: t={float(solution['motion_time']):.6f}s, jerk_rms={jerk_rms_pct:.6f}\n"
        f"SPCTOM: t={float(solution_spctom['motion_time']):.6f}s, jerk_rms={jerk_rms_sp:.6f}",
        va='top', ha='left', fontsize=11,
    )

    fig_cmp.tight_layout(rect=(0, 0.02, 1, 0.96))

    plt.show()
if __name__ == "__main__":
    main()