"""
Implements an adapted version of smooth path-constrained time-optimal motion planning (SPCTOM).

"""

# Imports
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Any


class SPCTOM:
    """
    Class for path-constrained time-optimal motion planning (PCTOM).
    """
    def __init__(self, robot=None, path_param=None, s_values=None,
                 T_max=None, T_min=None, T_dot_max=None, T_dot_min=None,
                 boundary_conditions=None, ds=None):
        # Keep explicit attrs for static analyzers and safe defaults.
        self.robot: Any = None
        self.path_param: Any = None
        self.s_values: Any = None
        self.ds = ds
        self.boundary_conditions: Any = None
        self.T_max: Any = None
        self.T_min: Any = None
        self.T_dot_max: Any = None
        self.T_dot_min: Any = None
        self.q_cache = {}
        self.T_coeffs_cache = {}
        self.T_dot_coeffs_cache = {}
        self.s_knots: Any = None
        self.v_init: Any = None
        self.v_bounds: Any = None

        # Optional direct construction in one call.
        if robot is not None and path_param is not None and s_values is not None:
            self.setup(
                robot=robot,
                path_param=path_param,
                s_values=s_values,
                T_max=T_max,
                T_min=T_min,
                T_dot_max=T_dot_max,
                T_dot_min=T_dot_min,
                boundary_conditions=boundary_conditions,
            )
    def q_s(self, s, q_prev=None):
        """
        Compute joint configuration q(s) along the path for path parameter s.
        """
        r_s = self.path_param.compute_r_s(s)
        if q_prev is None:
            # Reuse the nearest cached IK solution when available; otherwise
            # fall back to a neutral seed. This is more stable than always
            # starting from zeros for position-only IK.
            if self.q_cache:
                nearest_s = min(self.q_cache.keys(), key=lambda s0: abs(float(s0) - float(s)))
                q_prev = self.q_cache[nearest_s]
            else:
                q_prev = np.zeros(self.robot.n_joints())
        q = self.robot.inverse_kinematics(r_s, q0=q_prev)
        self.q_cache[s] = q
        return q
    
    def dq_ds(self, s, q_s):
        """
        Compute joint velocity dq/ds at path parameter s given q(s).
        Uses the position rows of the Jacobian (first 3) and pinv for the
        minimum-norm solution (handles redundancy in Planar3R).
        """
        J_pos = self.robot.jacobian(q_s)[:3, :]
        dr_ds = self.path_param.compute_dr_ds(s)
        return np.linalg.pinv(J_pos) @ dr_ds
    
    def d2q_ds2(self, s, q_s, dq_ds):
        """
        Compute joint acceleration d²q/ds² at path parameter s given q(s) and dq/ds.
        Uses numerical dJ/ds and pinv for minimum-norm solution.
        """
        J_pos   = self.robot.jacobian(q_s)[:3, :]              # 3×n
        dJ_ds   = self.robot.jacobian_derivative(q_s, dq_ds)[:3, :]   # 3×n
        d2r_ds2 = self.path_param.compute_d2r_ds2(s)              # 3-vector
        rhs = d2r_ds2 - dJ_ds @ dq_ds
        return np.linalg.pinv(J_pos) @ rhs
    
    def get_T(self, q, dq, ddq):
        q = np.asarray(q).flatten()
        dq = np.asarray(dq).flatten()
        ddq = np.asarray(ddq).flatten()
        return self.robot.torque(q, dq, ddq)
    
    def get_T_coeffs(self, s):
        '''
        Return the coefficients a(s), b(s), c(s) for the torque equation:
        T = a(s)·s̈ + b(s)·ṡ² + c(s)
        where:
        a(s) = M(q).q'
        b(s) = M(q).q'' + q'T C(q, q)
        c(s) = G(q)
        '''
        q = self.q_s(s)
        qp = self.dq_ds(s, q)
        qpp = self.d2q_ds2(s, q, qp)
        # Get params
        M = self.robot.mass_matrix(q)
        # C = self.robot.coriolis(q, qp)
        G = self.robot.gravity(q)
        a_s = M @ qp
        # b_s = M @ qpp + C @ qp
        c_s = G
        # Backend-agnostic computation for b(s):
        # at sdot=1, sddot=0 => dq = qp, ddq = qpp and T = b + c
        T_unit = self.robot.torque(q, qp, qpp)
        b_s = T_unit - c_s
        return a_s, b_s, c_s

    def get_T_dot(self, q, dq, ddq, dddq, dt=1e-6):
        '''
        Computes troque rate numerically using forward Euler.
        Analytical computation from thrd-order dynamics increases complexity with robot matrix derivatives
        '''
        q_next = q + dq * dt
        dq_next = dq + ddq * dt
        ddq_next = ddq + dddq * dt
        T1 = self.get_T(q, dq, ddq)
        T2 = self.get_T(q_next, dq_next, ddq_next)
        return (T2 - T1) / dt
    
    def get_T_dot_coeffs(self, s):
        '''
        Return the coefficients a(s), b(s), c(s), d(s) for the torque rate equation:
        Ṫ = a(s)·s̈̇ + b(s)·ṡ·s̈ + c(s)·ṡ³ + d(s)·ṡ
        where:
        a(s) = M.q'
        b(s) = 3.M.q'' + dM/ds.q' + 2.q'T.C.q'
        c(s) = M.q''\' + dM/ds.q'' + q''T.C.q' + q'T.dC/ds.q' + q'T.C.q''
        d(s) = dG/ds
        '''
        q = self.q_s(s)
        qp = self.dq_ds(s, q)
        qpp = self.d2q_ds2(s, q, qp)
        
        # We find the coeffs via probing the polynomail (numerical)
        # Helper
        def T_dot_eval(sdot, sddot, sdddot):
            dq = qp * sdot
            ddq = qp * sddot + qpp * sdot**2
            dddq = qp * sdddot + 3 * qpp * sdot * sddot # The q''' term is often ignored to simplfiy complexity. Its value is usually lower.
            return self.get_T_dot(q, dq, ddq, dddq)
        
        a_s = T_dot_eval(0, 0, 1)
        c_s = (T_dot_eval(2, 0, 0) - 2 * T_dot_eval(1, 0, 0)) / 6
        d_s = T_dot_eval(1, 0, 0) - c_s
        b_s = T_dot_eval(1, 1, 0) - d_s - c_s
        return a_s, b_s, c_s, d_s
    
    def _sddot_bounds_from_T(self, a_s, b_s, c_s, tau_min, tau_max, sdot):
        """
        Compute sddot bounds for given sdot from torque constraint
        """
        sddot_bounds = []

        for i in range(len(a_s)): # Across joints
            ai, bi, ci = a_s[i], b_s[i], c_s[i]

            if abs(ai) < 1e-8:
                # If ai is zero, the constraint is independent of sddot. Check if it's satisfied.
                tau_i = bi * sdot**2 + ci
                if tau_i < tau_min[i] or tau_i > tau_max[i]:
                    return np.nan, np.nan  # No feasible sddot
                else:
                    continue  # This joint does not constrain sddot

            # Compute bounds
            ub = (tau_max[i] - bi * sdot**2 - ci) / ai
            lb = (tau_min[i] - bi * sdot**2 - ci) / ai

            if ai > 0:
                sddot_bounds.append((lb, ub))
            else:
                # flip inequalities
                sddot_bounds.append((ub, lb))

        sddot_min = max([bound[0] for bound in sddot_bounds])
        sddot_max = min([bound[1] for bound in sddot_bounds])

        return sddot_min, sddot_max

    def _sdot_max_from_T(self, a_s, b_s, c_s, tau_min, tau_max, sdot_upper=100.0, tol=1e-6):
        """
        The maximum sdot is the largest value such that sddot_min <= sddot_max. We can find it by scanning sdot and checking feasibility of sddot bounds via binary search. Could be brittle in the sense i have to put an upper bound and tolerance, but it is more robust than trying to solve the quartic equation for the intersection of the constraints.
        """
        low = 0.0
        high = sdot_upper

        def is_feasible(sdot):
            sddot_min, sddot_max = self._sddot_bounds_from_T(a_s, b_s, c_s, tau_min, tau_max, sdot)
            return sddot_min <= sddot_max

        # binary search
        for _ in range(100):
            mid = 0.5 * (low + high)
            if is_feasible(mid):
                low = mid
            else:
                high = mid
        return low

    def _sdddot_bounds_from_Tdot(self, a_s, b_s, c_s, d_s, tdot_min, tdot_max, sddot, sdot):
        """
        Compute sdddot bounds from torque rate constraints given sddot and sdot.
        Tdot_min <= a sdddot + b sdot sddot + c sdot^3 + d sdot <= Tdot_max
        """
        sdddot_bounds = []
        for i in range(len(a_s)): # Across joints
            ai, bi, ci, di = a_s[i], b_s[i], c_s[i], d_s[i]

            if abs(ai) < 1e-8:
                # If ai is zero, the constraint is independent of sdddot. Check if it's satisfied.
                tdot_i = bi * sdot * sddot + ci * sdot**3 + di * sdot
                if tdot_i < tdot_min[i] or tdot_i > tdot_max[i]:
                    return np.nan, np.nan  # No feasible sdddot
                else:
                    continue  # This joint does not constrain sddot

            # Compute bounds
            ub = (tdot_max[i] - bi * sdot * sddot - ci * sdot**3 - di * sdot) / ai
            lb = (tdot_min[i] - bi * sdot * sddot - ci * sdot**3 - di * sdot) / ai

            if ai > 0:
                sdddot_bounds.append((lb, ub))
            else:
                # flip inequalities
                sdddot_bounds.append((ub, lb))

        sdddot_min = max([bound[0] for bound in sdddot_bounds])
        sdddot_max = min([bound[1] for bound in sdddot_bounds])

        return sdddot_min, sdddot_max
    
    def _is_feasible_T_and_Tdot(self, a_T, b_T, c_T, a_dT, b_dT, c_dT, d_dT, tau_min, tau_max, tdot_min, tdot_max, sdot):
        """
        Check if there exists a feasible sddot that satisfies both torque and torque-rate constraints at given sdot.
        """

        sddot_min, sddot_max = self._sddot_bounds_from_T(
            a_T, b_T, c_T, tau_min, tau_max, sdot
        )

        if not (sddot_min <= sddot_max):
            return False

        # sample a few candidate accelerations; how many to sample is a tradeoff between accuracy and speed. In practice, the bounds often collapse quickly as sdot increases, so we don't need many samples to check feasibility. TODO

        for alpha in np.linspace(0.0, 1.0, 5):
            sddot = sddot_min + alpha * (sddot_max - sddot_min)

            sdddot_min, sdddot_max = self._sdddot_bounds_from_Tdot(a_dT, b_dT, c_dT, d_dT, tdot_min, tdot_max, sddot, sdot)

            if sdddot_min <= sdddot_max:
                return True

        return False

    def _sdot_max_from_T_and_Tdot(self, a_T, b_T, c_T, a_dT, b_dT, c_dT, d_dT,
                                tau_min, tau_max, tdot_min, tdot_max,
                                sdot_upper=1.0, max_expand=8):

        def is_feasible(sdot):
            return self._is_feasible_T_and_Tdot(a_T, b_T, c_T, a_dT, b_dT, c_dT, d_dT, tau_min, tau_max, tdot_min, tdot_max, sdot)

        low = 0.0
        high = float(max(sdot_upper, 1e-3))

        # Expand upper bound until infeasible (or until max_expand attempts)
        for _ in range(max_expand):
            if is_feasible(high):
                low = high
                high *= 2.0
            else:
                break

        # binary search
        for _ in range(100):
            mid = 0.5 * (low + high)
            if is_feasible(mid):
                low = mid
            else:
                high = mid
        return low

    def compute_tau_s(self, s, sdot, sddot):
        A, B, C = self.get_T_coeffs(s)
        return A * sddot + B * sdot**2 + C
    
    def compute_tau_dot_s(self, s, sdot, sddot, sdddot):
        a_s, b_s, c_s, d_s = self.get_T_dot_coeffs(s)
        return a_s * sdddot + b_s * sdot * sddot + c_s * sdot**3 + d_s * sdot
    
    def setup(self, robot, path_param, s_values, T_max=None, T_min=None, T_dot_max=None, T_dot_min=None, boundary_conditions=None):
        """
        Setup planner for given problem
        """

        # Pre-inits
        self.robot = robot

        # Objective will be setup by passing to solve()

        # Boundary conditions
        self.boundary_conditions = boundary_conditions

        # path constraint - path setup by other class PathParametrize. Here only initialize
        self.s_values = s_values
        self.path_param = path_param

        # Setup torque and torque-rate limits
        self.T_max = T_max
        self.T_min = T_min
        self.T_dot_max = T_dot_max
        self.T_dot_min = T_dot_min
        # If torque limits are not provided, assume infinite (unconstrained)
        if self.T_max is None:
            self.T_max = np.ones(self.robot.n_joints()) * np.inf
        if self.T_min is None:
            self.T_min = -np.ones(self.robot.n_joints()) * np.inf
        if self.T_dot_max is None:
            self.T_dot_max = np.ones(self.robot.n_joints()) * np.inf
        if self.T_dot_min is None:
            self.T_dot_min = -np.ones(self.robot.n_joints()) * np.inf
        self.T_max = np.asarray(self.T_max, dtype=float)
        self.T_min = np.asarray(self.T_min, dtype=float)
        self.T_dot_max = np.asarray(self.T_dot_max, dtype=float)
        self.T_dot_min = np.asarray(self.T_dot_min, dtype=float)
        
        # Misc.
        self.q_cache = {}  # Cache for q(s) to avoid redundant IK calls


        # Get T dot coeffs
        self.T_dot_coeffs_cache = {}
        for s in s_values:
            self.T_dot_coeffs_cache[float(s)] = self.get_T_dot_coeffs(float(s))
        
        # Get T coeffs
        self.T_coeffs_cache = {}
        for s in s_values:
            self.T_coeffs_cache[float(s)] = self.get_T_coeffs(float(s))

        # Actual setup defaults for optimisation variables.
        self._set_default_knots_and_guess(num_knots=20)

    def _set_default_knots_and_guess(self, num_knots=20):
        """Fallback uniform knots + conservative initial speed profile."""
        self.s_knots = np.linspace(self.s_values[0], self.s_values[-1], int(num_knots))
        self.v_init = np.ones_like(self.s_knots, dtype=float) * 0.1
        self.v_init[0] = 1e-3
        self.v_init[-1] = 1e-3
        self.v_bounds = [(1e-6, None) for _ in self.s_knots]

    def warm_start_from_pctom(self, pctom_result, n_uniform_fill=4, boundary_pad=1, min_knots=12):
        """
        Warm-start SPCTOM from PCTOM trajectory:
        - knot locations from switching points + uniform fill
        - initial knot speeds by interpolation of PCTOM sdot(s)
        """
        s_knots = build_knots_from_pctom(
            pctom_result,
            self.s_values,
            n_uniform_fill=n_uniform_fill,
            boundary_pad=boundary_pad,
        )

        # Ensure enough DOFs for the spline optimizer.
        if len(s_knots) < int(min_knots):
            s0, sf = float(self.s_values[0]), float(self.s_values[-1])
            extra = np.linspace(s0, sf, int(min_knots))
            s_knots = np.unique(np.concatenate([s_knots, extra]))

        s_pct = np.asarray(pctom_result['s_traj'], dtype=float)
        v_pct = np.asarray(pctom_result['sdot_traj'], dtype=float)
        order = np.argsort(s_pct)
        s_pct = s_pct[order]
        v_pct = np.maximum(v_pct[order], 1e-9)

        # Deduplicate s before interpolation.
        s_u, idx_u = np.unique(np.round(s_pct, 12), return_index=True)
        v_u = v_pct[idx_u]

        self.s_knots = s_knots
        self.v_init = np.interp(self.s_knots, s_u, v_u, left=v_u[0], right=v_u[-1])
        self.v_init = np.maximum(self.v_init, 1e-6)
        self.v_init[0] = max(float(v_u[0]), 1e-6)
        self.v_init[-1] = max(float(v_u[-1]), 1e-6)
        self.v_bounds = [(1e-6, None) for _ in self.s_knots]

    def _build_spline(self, sdot_knots):
        return CubicSpline(self.s_knots, sdot_knots, bc_type='clamped')

    def solve(self, max_iter=1000, phi0=0.2, k_phi=0.5, tol=1e-6,
            alpha=1.0, beta=0.5, gamma=2.0, verbose=False, max_wall_time_s=20.0):
        """
        Solve SPCTOM using the Flexible Tolerance Method (FTM).

        Outer optimisation : Flexible Polyhedron Method (Nelder-Mead), Eq. A-4/A-5
        Inner optimisation : bisection line-search to recover feasibility, Fig. A.1
        Tolerance criterion: Eq. A-17, initial value phi0 = 0.2

        Parameters
        ----------
        max_iter : int   – maximum FPM iterations
        phi0     : float – initial near-feasibility tolerance  (Φ⁰, paper sets 0.2)
        k_phi    : float – scale factor for Φ update           (Eq. A-17)
        tol      : float – convergence threshold on f-spread
        alpha    : float – reflection coefficient  (standard NM = 1.0)
        beta     : float – contraction coefficient (standard NM = 0.5)
        gamma    : float – expansion coefficient   (standard NM = 2.0)
        verbose  : bool  – print iteration info

        Returns
        -------
        dict with keys:
            'sdot_knots'  – optimal sdot at each knot
            'spline'      – CubicSpline of sdot(s)
            'motion_time' – total trajectory time  (objective value)
            'iterations'  – number of FPM iterations executed
            'phi_final'   – final tolerance criterion value
        """
        import numpy as np
        import time

        t_start = time.perf_counter()

        n  = len(self.s_knots)           # dimension of search space
        bc = (self.v_init[0], self.v_init[-1])   # boundary sdots (fixed throughout)

        # ─── helpers to enforce boundary conditions ────────────────────────────
        def _apply_bc(x):
            x = x.copy()
            x[0]  = bc[0]
            x[-1] = bc[1]
            return np.maximum(x, 1e-9)

        # ─── objective  f(x) : total motion time  ─────────────────────────────
        _s_quad = np.linspace(self.s_knots[0], self.s_knots[-1], 400)
        s_all = np.asarray(self.s_values, dtype=float)
        if s_all.size > 201:
            idx = np.linspace(0, s_all.size - 1, 201, dtype=int)
            s_check = s_all[idx]
        else:
            s_check = s_all
        coeff_T = [self.T_coeffs_cache[float(s)] for s in s_check]
        coeff_Tdot = [self.T_dot_coeffs_cache[float(s)] for s in s_check]
        s_all = np.asarray(self.s_values, dtype=float)
        if s_all.size > 201:
            idx = np.linspace(0, s_all.size - 1, 201, dtype=int)
            s_check = s_all[idx]
        else:
            s_check = s_all

        def objective(x):
            sp = self._build_spline(_apply_bc(x))
            sdot = np.maximum(sp(_s_quad), 1e-9)
            trapz_fn = getattr(np, 'trapezoid', np.trapz)
            return float(trapz_fn(1.0 / sdot, _s_quad))

        # ─── constraint-violation measure  T(x)  Eq. A-15 ────────────────────
        def constraint_violation(x, stop_at=None):
            sp = self._build_spline(_apply_bc(x))

            # --- Evaluate spline and derivatives ---
            sdot = np.maximum(sp(s_check), 1e-9)          # (K,)
            dsdot_ds = sp(s_check, 1)                     # (K,)
            d2sdot_ds2 = sp(s_check, 2)                   # (K,)

            # --- Convert to time derivatives ---
            sddot = dsdot_ds * sdot                       # (K,)
            sdddot = sdot * (dsdot_ds**2 + sdot * d2sdot_ds2)  # (K,)

            vio = 0.0

            # =========================
            # Torque constraints
            # =========================
            # coeff_T: shape (K, n_joints, 3)
            a_T = np.array([coeff[0] for coeff in coeff_T])  # (K, n_joints)
            b_T = np.array([coeff[1] for coeff in coeff_T])  # (K, n_joints)
            c_T = np.array([coeff[2] for coeff in coeff_T])  # (K, n_joints)

            tau = (
                a_T * sddot[:, None]
                + b_T * (sdot[:, None] ** 2)
                + c_T
            )  # (K, n_joints)

            # Upper limits
            mask_max = (self.T_max < np.inf) & (self.T_max > 0)
            if np.any(mask_max):
                v_max = tau[:, mask_max] / self.T_max[mask_max] - 1.0
                vio = max(vio, np.max(v_max))

            # Lower limits
            mask_min = (self.T_min > -np.inf) & (self.T_min < 0)
            if np.any(mask_min):
                v_min = tau[:, mask_min] / self.T_min[mask_min] - 1.0
                vio = max(vio, np.max(v_min))

            if stop_at is not None and vio > stop_at:
                return vio

            # =========================
            # Torque derivative constraints
            # =========================
            # coeff_Tdot: shape (K, n_joints, 4)
            a_dT = np.array([coeff[0] for coeff in coeff_Tdot])  # (K, n_joints)
            b_dT = np.array([coeff[1] for coeff in coeff_Tdot])
            c_dT = np.array([coeff[2] for coeff in coeff_Tdot])  # (K, n_joints)
            d_dT = np.array([coeff[3] for coeff in coeff_Tdot])

            tdot = (
                a_dT * sdddot[:, None]
                + b_dT * (sdot[:, None] * sddot[:, None])
                + c_dT * (sdot[:, None] ** 3)
                + d_dT * sdot[:, None]
            )  # (K, n_joints)

            # Upper limits
            mask_max = (self.T_dot_max < np.inf) & (self.T_dot_max > 0)
            if np.any(mask_max):
                v_max = tdot[:, mask_max] / self.T_dot_max[mask_max] - 1.0
                vio = max(vio, np.max(v_max))

            # Lower limits
            mask_min = (self.T_dot_min > -np.inf) & (self.T_dot_min < 0)
            if np.any(mask_min):
                v_min = tdot[:, mask_min] / self.T_dot_min[mask_min] - 1.0
                vio = max(vio, np.max(v_min))

            return vio

        # ─── near-feasibility test ────────────────────────────────────────────
        def is_acceptable(x, phi):
            return constraint_violation(x, stop_at=phi) <= phi

        # ─── inner optimisation: bisection line-search  Fig. A.1 ──────────────
        def bisect_to_feasible(x_bad, x_anchor, phi, max_steps=25):
            lo, hi = 0.0, 1.0
            for _ in range(max_steps):
                mid   = 0.5 * (lo + hi)
                x_mid = _apply_bc(x_bad + mid * (x_anchor - x_bad))
                if is_acceptable(x_mid, phi):
                    hi = mid
                else:
                    lo = mid
            return _apply_bc(x_bad + hi * (x_anchor - x_bad))

        # ─── initialise polyhedron  (n+1 vertices in R^n) ─────────────────────
        x0    = _apply_bc(np.array(self.v_init, dtype=float))
        x_slow = _apply_bc(np.full(n, 1e-4))   # very slow → always feasible

        verts = np.empty((n + 1, n))
        verts[0] = x0
        for i in range(n):
            xi    = x0.copy()
            if 0 < i < n - 1:
                xi[i] *= 1.05
            verts[i + 1] = _apply_bc(xi)

        phi    = phi0
        f_vals = np.array([objective(v) for v in verts])
        T_vals = np.array([constraint_violation(v) for v in verts])

        # push any nonfeasible initial vertices toward slow-motion origin
        for j in range(n + 1):
            if T_vals[j] > phi:
                verts[j]  = bisect_to_feasible(verts[j], x_slow, phi)
                f_vals[j] = objective(verts[j])
                T_vals[j] = constraint_violation(verts[j])

        best_idx = int(np.argmin(f_vals))
        best_x   = verts[best_idx].copy()
        best_f   = f_vals[best_idx]

        it = -1
        for it in range(max_iter):
            if (time.perf_counter() - t_start) > max_wall_time_s:
                if verbose:
                    print(f"[FTM] Stopping at iter {it} due to wall-time limit ({max_wall_time_s:.1f}s)")
                break

            # ── FIX 1: check all points acceptable at top of every iteration ──
            # Mirrors Fig. A.2 top: "All points acceptable? No → move infeasible
            # points toward origin."  In the paper the "origin" is the slow-motion
            # point (very small sdot), which is always feasible.
            for j in range(n + 1):
                if not is_acceptable(verts[j], phi):
                    verts[j]  = bisect_to_feasible(verts[j], x_slow, phi)
                    f_vals[j] = objective(verts[j])
                    T_vals[j] = constraint_violation(verts[j])

            # ── sort: verts[0] = best (lowest f), verts[-1] = worst ──────────
            order  = np.argsort(f_vals)
            verts  = verts[order]
            f_vals = f_vals[order]
            T_vals = T_vals[order]

            x_best  = verts[0]
            x_worst = verts[-1]

            # centroid of all but worst
            x_cen = _apply_bc(verts[:-1].mean(axis=0))

            # ── FIX 2: convergence uses polyhedron size (geometric), not just
            # f-spread.  The flowchart checks "polyhedron too small?" which
            # corresponds to the diameter of the simplex in decision space.
            # We keep f_spread as a secondary guard but make the primary check
            # the maximum distance between any vertex and the centroid —
            # i.e. the geometric size of the polyhedron.
            poly_size = float(np.max(np.linalg.norm(verts - x_cen, axis=1)))
            f_spread  = f_vals[-1] - f_vals[0]
            if poly_size < tol and T_vals[0] <= phi:
                if verbose:
                    print(f"[FTM] Converged at iter {it}  f={f_vals[0]:.6f}  "
                            f"poly_size={poly_size:.2e}  Φ={phi:.2e}")
                break

            if verbose and it % 50 == 0:
                print(f"[FTM] iter {it:4d}  f_best={f_vals[0]:.5f}  "
                    f"f_spread={f_spread:.2e}  poly_size={poly_size:.2e}  "
                    f"Φ={phi:.2e}  T_best={T_vals[0]:.2e}")

            # ── reflection ────────────────────────────────────────────────────
            x_ref = _apply_bc(x_cen + alpha * (x_cen - x_worst))

            if not is_acceptable(x_ref, phi):
                x_ref = bisect_to_feasible(x_ref, x_cen, phi)

            f_ref = objective(x_ref)
            T_ref = constraint_violation(x_ref)

            if f_ref < f_vals[0]:
                # reflection beat best → try expansion
                x_exp = _apply_bc(x_cen + gamma * (x_cen - x_worst))

                if not is_acceptable(x_exp, phi):
                    x_exp = bisect_to_feasible(x_exp, x_ref, phi)

                f_exp = objective(x_exp)
                T_exp = constraint_violation(x_exp)

                if f_exp < f_ref:
                    verts[-1]  = x_exp
                    f_vals[-1] = f_exp
                    T_vals[-1] = T_exp
                else:
                    verts[-1]  = x_ref
                    f_vals[-1] = f_ref
                    T_vals[-1] = T_ref

            elif f_ref < f_vals[-2]:
                # reflection is better than second-worst → accept
                verts[-1]  = x_ref
                f_vals[-1] = f_ref
                T_vals[-1] = T_ref

            else:
                # ── contraction ───────────────────────────────────────────────
                x_con = _apply_bc(x_cen + beta * (x_worst - x_cen))

                if not is_acceptable(x_con, phi):
                    x_con = bisect_to_feasible(x_con, x_cen, phi)

                f_con = objective(x_con)
                T_con = constraint_violation(x_con)

                if f_con < f_vals[-1]:
                    verts[-1]  = x_con
                    f_vals[-1] = f_con
                    T_vals[-1] = T_con
                else:
                    # ── FIX 3 (collapse): pull all non-best vertices toward
                    # x_best only — no special x_slow bisection during collapse.
                    # Fig. A.1(d) specifies shrinking toward the best vertex;
                    # feasibility is guaranteed by the per-iteration check at the
                    # top of the next iteration (Fix 1), not here.
                    for j in range(1, n + 1):
                        verts[j]  = _apply_bc(x_best + beta * (verts[j] - x_best))
                        f_vals[j] = objective(verts[j])
                        T_vals[j] = constraint_violation(verts[j])

            # track global best (must be acceptable at current Φ)
            for j in range(n + 1):
                if f_vals[j] < best_f and T_vals[j] <= phi:
                    best_f = f_vals[j]
                    best_x = verts[j].copy()

            # ── FIX 4: update Φ at the END of the iteration (k=k+1 in Fig. A.3)
            # Previously this was done near the top, before operations.
            # The flowchart places it after all moves, before looping back.
            phi_a = k_phi * float(np.max(np.linalg.norm(verts - x_cen, axis=1)))
            phi_b = k_phi * float(np.linalg.norm(x_cen - x_worst))
            phi   = min(phi, phi_a, phi_b)
            phi   = max(phi, tol * 1e-2)   # floor to avoid premature termination

        best_x      = _apply_bc(best_x)
        best_spline = self._build_spline(best_x)

        return {
            'sdot_knots':  best_x,
            'spline':      best_spline,
            'motion_time': best_f,
            'iterations':  it + 1,
            'phi_final':   phi,
        }

    def plot_solution(self, solution, s_eval=None, ax=None, label='SPCTOM'):
        """
        Plot SPCTOM speed profile sdot(s) from the optimized spline.
        """
        import matplotlib.pyplot as plt

        if s_eval is None:
            s_eval = np.asarray(self.s_values)
        else:
            s_eval = np.asarray(s_eval)

        spline = solution['spline']
        sdot = np.maximum(spline(s_eval), 1e-9)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure

        ax.plot(s_eval, sdot, linewidth=2.0, label=label)
        ax.set_xlabel('s')
        ax.set_ylabel('sdot')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        return fig, ax
    

# ─────────────────────────────────────────────────────────────────────────────
# Knot-point builder
# ─────────────────────────────────────────────────────────────────────────────

def build_knots_from_pctom(pctom_result, s_values,
                            n_uniform_fill=0,
                            boundary_pad=0):
    """
    Build SPCTOM knot s-values from PCTOM switching points.

    Strategy
    --------
    1.  Always include s_start and s_end.
    2.  Include every PCTOM switching point.
    3.  Between consecutive switching points, insert n_uniform_fill evenly
        spaced interior knots so the spline also has support in smooth regions.
    4.  Optionally add boundary_pad knots just inside each end to let the
        spline match the zero-velocity boundary conditions smoothly.

    Parameters
    ----------
    pctom_result     : dict returned by PCTOMSolver.solve()
    s_values         : full s-grid (used for clipping)
    n_uniform_fill   : extra knots between each pair of switching points
    boundary_pad     : number of knots to add near each boundary

    Returns
    -------
    knots : sorted 1-D numpy array of s knot values
    """
    s0 = s_values[0]
    sf = s_values[-1]
    sw = pctom_result['switching_s']

    # Anchor points: boundaries + switching points
    anchors = np.concatenate([[s0], sw, [sf]])
    anchors = np.clip(anchors, s0, sf)
    anchors = np.unique(anchors)

    # Fill between anchors
    all_knots = list(anchors)
    for i in range(len(anchors) - 1):
        if n_uniform_fill > 0:
            fill = np.linspace(anchors[i], anchors[i + 1],
                               n_uniform_fill + 2)[1:-1]
            all_knots.extend(fill.tolist())

    # Boundary padding
    span = sf - s0
    for k in range(1, boundary_pad + 1):
        frac = k / (boundary_pad + 1) * 0.05  # within first/last 5% of path
        all_knots.append(s0 + frac * span)
        all_knots.append(sf - frac * span)

    knots = np.unique(np.clip(all_knots, s0, sf))
    return knots
