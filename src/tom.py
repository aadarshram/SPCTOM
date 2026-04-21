"""
Base class for Time-Optimal Motion (TOM) solver.
"""

# Imports
import numpy as np

class TOM:
    def __init__(self, path_param, robot, s_values, T_max=None, T_min=None, T_dot_max=None, T_dot_min=None, ds=None, boundary_conditions=None):
        self.path_param = path_param
        self.robot = robot
        self.s_values = s_values
        self.q_cache = {}
        self.s_values   = np.asarray(s_values, dtype=float)
        n = robot.n_joints()
        self.T_max = np.asarray(T_max, dtype=float) if T_max is not None \
                     else np.full(n,  np.inf)
        self.T_min = np.asarray(T_min, dtype=float) if T_min is not None \
                     else np.full(n, -np.inf)
        self.ds = ds if ds is not None \
                  else (s_values[-1] - s_values[0]) / max(len(s_values) - 1, 500)
        # Boundary conditions
        qdot0, qdotf = boundary_conditions['q_dot'] if boundary_conditions and 'q_dot' in boundary_conditions else (np.zeros(n), np.zeros(n))
        # Convert qdot boundary conditions to sdot boundary conditions using the path and robot kinematics at the start and end of the path
        sdot0 = self.sdot_from_qdot(s_values[0], self.q_s(s_values[0]), qdot0)
        sdotf = self.sdot_from_qdot(s_values[-1], self.q_s(s_values[-1]), qdotf)
        self.sdot0 = sdot0
        self.sdotf = sdotf
        # Find VLC
        self.sdot_max_values = self._compute_VLC()

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

    def sdot_from_qdot(self, s, q_s, dq):
        """
        Compute sdot from qdot at path parameter s given q(s) and dq.
        """
        qp = np.asarray(self.dq_ds(s, q_s), dtype=float).reshape(-1)
        dq = np.asarray(dq, dtype=float).reshape(-1)

        # qdot = qp * sdot  -> least-squares scalar estimate
        denom = float(qp @ qp)
        if denom < 1e-12:
            return 0.0
        return float((qp @ dq) / denom)
        
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
    
    def _compute_VLC(self):
        """
        Find Ṡ_max(s) across s based on torque constraints.
        """
        vlc = np.array([
            self._sdot_max_from_T(
                *self.get_T_coeffs(s),
                self.T_min, self.T_max
            ) for s in self.s_values
        ])
        return vlc

    def _sdot_max_at(self, s):
        """
        Get Ṡ_max at arbitrary s by interpolation.
        """
        return np.interp(s, self.s_values, self.sdot_max_values)
    
