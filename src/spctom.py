"""
Implements an adapted version of smooth path-constrained time-optimal motion planning (SPCTOM).

"""

# Imports
import numpy as np
from scipy.interpolate import CubicSpline

from src.tom import TOM
    

class SPCTOM(TOM):
    """
    Class for path-constrained time-optimal motion planning (PCTOM).
    """
    def __init__(self, robot=None, path_param=None, s_values=None,
                 T_max=None, T_min=None, T_dot_max=None, T_dot_min=None,
                 boundary_conditions=None, ds=None):
        self.T_coeffs_cache = {}
        self.T_dot_coeffs_cache = {}
        super().__init__(
            path_param=path_param,
            robot=robot,
            s_values=s_values,
            T_max=T_max,
            T_min=T_min,
            T_dot_max=T_dot_max,
            T_dot_min=T_dot_min,
            boundary_conditions=boundary_conditions,
            ds=ds,
        )

        self._prime_dynamics_caches()
        self._set_default_knots_and_guess(num_knots=20)

    def _cache_key(self, s):
        return float(s)

    def _prime_dynamics_caches(self):
        for s in np.asarray(self.s_values, dtype=float):
            self.get_T_coeffs(float(s))
            self.get_T_dot_coeffs(float(s))

    def _get_T_coeffs_cached(self, s):
        key = self._cache_key(s)
        if key not in self.T_coeffs_cache:
            self.T_coeffs_cache[key] = super().get_T_coeffs(key)
        return self.T_coeffs_cache[key]

    def _get_T_dot_coeffs_cached(self, s):
        key = self._cache_key(s)
        if key not in self.T_dot_coeffs_cache:
            self.T_dot_coeffs_cache[key] = super().get_T_dot_coeffs(key)
        return self.T_dot_coeffs_cache[key]

    def get_T_coeffs(self, s):
        return self._get_T_coeffs_cached(s)

    def get_T_dot_coeffs(self, s):
        return self._get_T_dot_coeffs_cached(s)

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
        
        NOTE: If the PCTOM trajectory violates torque-rate constraints too severely,
        falls back to default conservative initialization to ensure a feasible starting point.
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
        v_init_candidate = np.interp(self.s_knots, s_u, v_u, left=v_u[0], right=v_u[-1])
        v_init_candidate = np.maximum(v_init_candidate, 1e-6)
        v_init_candidate[0] = max(float(v_u[0]), 1e-6)
        v_init_candidate[-1] = max(float(v_u[-1]), 1e-6)

        # Test if the interpolated profile is feasible enough
        # Sample the candidate spline and check torque-rate violations
        # Use 'natural' BC to avoid overshoot artifacts from PCTOM discontinuities
        # Avoid exact boundaries where path geometry may be singular
        sp_test = CubicSpline(self.s_knots, v_init_candidate, bc_type='natural')
        s_test = np.linspace(self.s_knots[0] + 0.01 * (self.s_knots[-1] - self.s_knots[0]),
                            self.s_knots[-1] - 0.01 * (self.s_knots[-1] - self.s_knots[0]),
                            50)
        sdot_test = np.maximum(sp_test(s_test), 1e-9)
        dsdot_ds_test = sp_test(s_test, 1)
        d2sdot_ds2_test = sp_test(s_test, 2)
        sddot_test = dsdot_ds_test * sdot_test
        sdddot_test = sdot_test * (dsdot_ds_test**2 + sdot_test * d2sdot_ds2_test)
        
        # Check Tdot constraint violation
        Tdot_vio_max = 0.0
        for i, s in enumerate(s_test):
            a_Tdot, b_Tdot, c_Tdot, d_Tdot = self.get_T_dot_coeffs(float(s))
            tau_dot_joint = (a_Tdot * sdddot_test[i] + 
                            b_Tdot * sdot_test[i] * sddot_test[i] +
                            c_Tdot * (sdot_test[i]**3) +
                            d_Tdot * sdot_test[i])
            for j in range(self.robot.n_joints()):
                if self.T_dot_max[j] < np.inf and self.T_dot_max[j] > 0:
                    v = tau_dot_joint[j] / self.T_dot_max[j] - 1.0
                    Tdot_vio_max = max(Tdot_vio_max, v)
                if self.T_dot_min[j] > -np.inf and self.T_dot_min[j] < 0:
                    v = tau_dot_joint[j] / self.T_dot_min[j] - 1.0
                    Tdot_vio_max = max(Tdot_vio_max, v)
        
        if Tdot_vio_max > 50.0:
            # PCTOM trajectory has severe T_dot violations.
            # Try to adjust by scaling down the profile while preserving structure.
            print(f"[warm_start_from_pctom] PCTOM trajectory has Tdot_violation={Tdot_vio_max:.1e} >> 50.0.")
            print(f"[warm_start_from_pctom] Attempting constrained adjustment to satisfy T_dot constraints...")
            
            adjusted = self._adjust_profile_for_tdot_constraints(
                v_init_candidate, self.s_knots, target_vio=1.0, max_iterations=30
            )
            if adjusted is not None:
                self.v_init = adjusted
                self.v_bounds = [(1e-6, None) for _ in self.s_knots]
                print(f"[warm_start_from_pctom] Successfully adjusted warm-start to satisfy T_dot constraints.")
            else:
                # If adjustment fails, fall back to conservative initialization
                print(f"[warm_start_from_pctom] Adjustment failed. Falling back to conservative default initialization.")
                self._set_default_knots_and_guess(num_knots=max(20, len(s_knots)))
        else:
            self.v_init = v_init_candidate
            self.v_bounds = [(1e-6, None) for _ in self.s_knots]

    def _measure_tdot_violation(self, v_knots, s_knots=None, n_samples=50):
        """
        Measure maximum T_dot constraint violation for a given speed profile.
        
        Use natural boundary conditions (second derivative = 0 at boundaries) to avoid
        oscillations. Avoid evaluating exactly at s=0 and s=1 boundaries where geometry
        may be singular.
        
        Returns: (max_violation, details_dict)
        """
        if s_knots is None:
            s_knots = self.s_knots
        
        # Use 'natural' boundary condition to avoid overshoot at boundaries
        sp = CubicSpline(s_knots, v_knots, bc_type='natural')
        # Avoid exact boundaries where path geometry may be singular
        s_test = np.linspace(s_knots[0] + 0.01 * (s_knots[-1] - s_knots[0]),
                            s_knots[-1] - 0.01 * (s_knots[-1] - s_knots[0]),
                            n_samples)
        sdot = np.maximum(sp(s_test), 1e-9)
        dsdot_ds = sp(s_test, 1)
        d2sdot_ds2 = sp(s_test, 2)
        sddot = dsdot_ds * sdot
        sdddot = sdot * (dsdot_ds**2 + sdot * d2sdot_ds2)
        
        max_vio = 0.0
        for i, s in enumerate(s_test):
            a_Tdot, b_Tdot, c_Tdot, d_Tdot = self.get_T_dot_coeffs(float(s))
            tau_dot_joint = (a_Tdot * sdddot[i] + 
                            b_Tdot * sdot[i] * sddot[i] +
                            c_Tdot * (sdot[i]**3) +
                            d_Tdot * sdot[i])
            for j in range(self.robot.n_joints()):
                if self.T_dot_max[j] < np.inf and self.T_dot_max[j] > 0:
                    v = tau_dot_joint[j] / self.T_dot_max[j] - 1.0
                    max_vio = max(max_vio, v)
                if self.T_dot_min[j] > -np.inf and self.T_dot_min[j] < 0:
                    v = tau_dot_joint[j] / self.T_dot_min[j] - 1.0
                    max_vio = max(max_vio, v)
        
        return max_vio, {'max_vio': max_vio, 'n_samples': n_samples}

    def _adjust_profile_for_tdot_constraints(self, v_init_candidate, s_knots, 
                                            target_vio=1.0, max_iterations=30):
        """
        Adjust PCTOM speed profile to satisfy T_dot constraints.
        
        Strategy: Binary search on a global scaling factor that reduces ALL speeds
        uniformly (including boundaries) while maintaining the relative profile shape.
        Find a scaling where T_dot violations are acceptable (near target_vio).
        
        Returns: Adjusted speed profile, or None if adjustment fails.
        """
        # Binary search on global scaling factor
        scale_min = 1e-5
        scale_max = 1.0
        best_scale = None
        best_vio = np.inf
        
        for iteration in range(max_iterations):
            scale_mid = 0.5 * (scale_min + scale_max)
            v_test = np.maximum(v_init_candidate * scale_mid, 1e-6)
            
            vio, _ = self._measure_tdot_violation(v_test, s_knots, n_samples=50)
            
            if abs(vio - target_vio) < best_vio:
                best_vio = abs(vio - target_vio)
                best_scale = scale_mid
            
            # If violation is acceptable (near 0 or within tolerance), accept this scale
            if vio <= target_vio:
                # Found a feasible solution, try to go higher
                scale_min = scale_mid
            else:
                # Too many violations, need to go lower
                scale_max = scale_mid
            
            # Check for convergence
            if abs(scale_max - scale_min) < 1e-8 or iteration > max_iterations - 1:
                break
        
        if best_scale is not None and best_scale > 1e-4:
            v_adjusted = np.maximum(v_init_candidate * best_scale, 1e-6)
            return v_adjusted
        
        return None

    def _build_spline(self, sdot_knots):
        return CubicSpline(self.s_knots, sdot_knots, bc_type='not-a-knot')

    def _check_and_fix_warm_start_feasibility(self, verbose=False):
        """
        Check if warm-start initial guess is feasible. If infeasible,
        find a uniform conservative speed that IS feasible.
        
        Returns: (did_modify, original_v_init, final_v_init)
        """
        # Build initial spline and sample it
        sp_test = self._build_spline(np.array(self.v_init, dtype=float))
        s_check = np.linspace(self.s_knots[0], self.s_knots[-1], 100)
        sdot = np.maximum(sp_test(s_check), 1e-9)
        sdot_deriv = sp_test(s_check, 1)
        sdot_deriv2 = sp_test(s_check, 2)
        sddot = sdot_deriv * sdot
        sdddot = sdot * (sdot_deriv**2 + sdot * sdot_deriv2)

        # Compute torque constraint violations
        T_vio = 0.0
        a_T = np.zeros((len(s_check), self.robot.n_joints()))
        b_T = np.zeros_like(a_T)
        c_T = np.zeros_like(a_T)
        for i, s in enumerate(s_check):
            a_T[i], b_T[i], c_T[i] = self.get_T_coeffs(float(s))
        tau = a_T * sddot[:, None] + b_T * (sdot[:, None]**2) + c_T
        for j in range(self.robot.n_joints()):
            if self.T_max[j] < np.inf:
                v = tau[:, j] / self.T_max[j] - 1.0
                T_vio = max(T_vio, np.max(v))
            if self.T_min[j] > -np.inf:
                v = tau[:, j] / self.T_min[j] - 1.0
                T_vio = max(T_vio, np.max(v))

        # Compute torque-rate constraint violations
        Tdot_vio = 0.0
        a_Tdot = np.zeros((len(s_check), self.robot.n_joints()))
        b_Tdot = np.zeros_like(a_Tdot)
        c_Tdot = np.zeros_like(a_Tdot)
        d_Tdot = np.zeros_like(a_Tdot)
        for i, s in enumerate(s_check):
            a_Tdot[i], b_Tdot[i], c_Tdot[i], d_Tdot[i] = self.get_T_dot_coeffs(float(s))
        tau_dot = (a_Tdot * sdddot[:, None] + 
                   b_Tdot * (sdot[:, None] * sddot[:, None]) +
                   c_Tdot * (sdot[:, None]**3) +
                   d_Tdot * sdot[:, None])
        for j in range(self.robot.n_joints()):
            if self.T_dot_max[j] < np.inf:
                v = tau_dot[:, j] / self.T_dot_max[j] - 1.0
                Tdot_vio = max(Tdot_vio, np.max(v))
            if self.T_dot_min[j] > -np.inf:
                v = tau_dot[:, j] / self.T_dot_min[j] - 1.0
                Tdot_vio = max(Tdot_vio, np.max(v))

        max_vio = max(T_vio, Tdot_vio)
        original_v_init = self.v_init.copy()

        if max_vio > 100.0:  # If massively infeasible
            if verbose:
                print(f"[SPCTOM] Warm-start is severely infeasible (max_vio={max_vio:.1e}). "
                      f"Finding feasible uniform scaling...")
            
            # Binary search for a uniform speed v such that v * v_init is feasible
            original_mean = np.mean(original_v_init[1:-1])  # Exclude boundaries
            v_max = min(1.0, original_mean)
            v_min = 1e-5
            
            for _ in range(20):
                v_mid = 0.5 * (v_min + v_max)
                v_test = original_v_init * v_mid
                v_test = np.maximum(v_test, 1e-6)
                v_test[0] = original_v_init[0] * v_mid
                v_test[-1] = original_v_init[-1] * v_mid
                
                sp_test_scaled = self._build_spline(v_test)
                sdot_test = np.maximum(sp_test_scaled(s_check), 1e-9)
                sdot_deriv_test = sp_test_scaled(s_check, 1)
                sdot_deriv2_test = sp_test_scaled(s_check, 2)
                sddot_test = sdot_deriv_test * sdot_test
                sdddot_test = sdot_test * (sdot_deriv_test**2 + sdot_test * sdot_deriv2_test)
                
                # Check feasibility of scaled guess
                vio_test = 0.0
                a_T_test = np.zeros((len(s_check), self.robot.n_joints()))
                b_T_test = np.zeros_like(a_T_test)
                c_T_test = np.zeros_like(a_T_test)
                for i, s in enumerate(s_check):
                    a_T_test[i], b_T_test[i], c_T_test[i] = self.get_T_coeffs(float(s))
                tau_test = a_T_test * sddot_test[:, None] + b_T_test * (sdot_test[:, None]**2) + c_T_test
                for j in range(self.robot.n_joints()):
                    if self.T_max[j] < np.inf:
                        v = tau_test[:, j] / self.T_max[j] - 1.0
                        vio_test = max(vio_test, np.max(v))
                    if self.T_min[j] > -np.inf:
                        v = tau_test[:, j] / self.T_min[j] - 1.0
                        vio_test = max(vio_test, np.max(v))
                
                a_Tdot_test = np.zeros((len(s_check), self.robot.n_joints()))
                b_Tdot_test = np.zeros_like(a_Tdot_test)
                c_Tdot_test = np.zeros_like(a_Tdot_test)
                d_Tdot_test = np.zeros_like(a_Tdot_test)
                for i, s in enumerate(s_check):
                    a_Tdot_test[i], b_Tdot_test[i], c_Tdot_test[i], d_Tdot_test[i] = self.get_T_dot_coeffs(float(s))
                tau_dot_test = (a_Tdot_test * sdddot_test[:, None] + 
                               b_Tdot_test * (sdot_test[:, None] * sddot_test[:, None]) +
                               c_Tdot_test * (sdot_test[:, None]**3) +
                               d_Tdot_test * sdot_test[:, None])
                for j in range(self.robot.n_joints()):
                    if self.T_dot_max[j] < np.inf:
                        v = tau_dot_test[:, j] / self.T_dot_max[j] - 1.0
                        vio_test = max(vio_test, np.max(v))
                    if self.T_dot_min[j] > -np.inf:
                        v = tau_dot_test[:, j] / self.T_dot_min[j] - 1.0
                        vio_test = max(vio_test, np.max(v))
                
                if vio_test < 1.0:  # Found feasible region
                    v_max = v_mid
                else:
                    v_min = v_mid
            
            # Use the best scaled version we found
            scale_final = v_max
            self.v_init = original_v_init * scale_final
            self.v_init = np.maximum(self.v_init, 1e-6)
            self.v_init[0] = original_v_init[0] * scale_final
            self.v_init[-1] = original_v_init[-1] * scale_final
            
            if verbose:
                print(f"[SPCTOM] Applied scaling factor {scale_final:.4f} to warm-start.")
            
            return True, original_v_init, self.v_init
        elif max_vio > 10.0:  # If moderately infeasible
            if verbose:
                print(f"[SPCTOM] Warm-start is moderately infeasible (max_vio={max_vio:.1e}). "
                      f"Applying 0.5x scaling.")
            
            # Scale down by 0.5
            scale_factor = 0.5
            self.v_init = original_v_init * scale_factor
            self.v_init = np.maximum(self.v_init, 1e-6)
            self.v_init[0] = original_v_init[0] * scale_factor
            self.v_init[-1] = original_v_init[-1] * scale_factor
            
            return True, original_v_init, self.v_init
        else:
            if verbose:
                print(f"[SPCTOM] Warm-start is feasible (max_vio={max_vio:.2e}).")
            return False, original_v_init, original_v_init


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
        coeff_T = [self._get_T_coeffs_cached(float(s)) for s in s_check]
        coeff_Tdot = [self._get_T_dot_coeffs_cached(float(s)) for s in s_check]

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
            # Always bisect toward a point that is known to be feasible at the
            # current tolerance. The centroid is not guaranteed to satisfy the
            # constraints, so fall back to the slow-motion anchor when needed.
            if not is_acceptable(x_anchor, phi):
                x_anchor = x_slow

            lo, hi = 0.0, 1.0
            best = _apply_bc(x_anchor)
            for _ in range(max_steps):
                mid   = 0.5 * (lo + hi)
                x_mid = _apply_bc(x_bad + mid * (x_anchor - x_bad))
                if is_acceptable(x_mid, phi):
                    hi = mid
                    best = x_mid
                else:
                    lo = mid
            return _apply_bc(best)

        # ─── initialise polyhedron  (n+1 vertices in R^n) ─────────────────────
        x0    = _apply_bc(np.array(self.v_init, dtype=float))
        x_slow = _apply_bc(np.full(n, 1e-7))   # ultra-slow anchor → must be feasible

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
        x_best   = best_x.copy()

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
                    verts[j]  = bisect_to_feasible(verts[j], x_best, phi)
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
                x_ref = bisect_to_feasible(x_ref, x_best, phi)

            f_ref = objective(x_ref)
            T_ref = constraint_violation(x_ref)

            if f_ref < f_vals[0]:
                # reflection beat best → try expansion
                x_exp = _apply_bc(x_cen + gamma * (x_cen - x_worst))

                if not is_acceptable(x_exp, phi):
                    x_exp = bisect_to_feasible(x_exp, x_best, phi)

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
                    x_con = bisect_to_feasible(x_con, x_best, phi)

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
            phi   = max(phi, tol * 10)   # floor at the requested tolerance scale

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
