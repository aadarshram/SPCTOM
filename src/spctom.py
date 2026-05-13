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
        return CubicSpline(self.s_knots, sdot_knots, bc_type='not-a-knot')

    def solve(self, max_iter=1000, phi0=None, k_phi=0.5, tol=1e-6,
          alpha=1.0, beta=0.5, gamma=2.0, verbose=False, max_wall_time_s=20.0):
        import time
        t_start = time.perf_counter()

        n = len(self.s_knots)

        # ── evaluation grids ─────────────────────────────────────────────────
        _s_quad = np.linspace(self.s_knots[0], self.s_knots[-1], 400)
        s_all   = np.asarray(self.s_values, dtype=float)
        idx     = np.linspace(0, s_all.size - 1, min(201, s_all.size), dtype=int)
        s_check = s_all[idx]

        coeff_T    = [self._get_T_coeffs_cached(float(s))     for s in s_check]
        coeff_Tdot = [self._get_T_dot_coeffs_cached(float(s)) for s in s_check]

        a_T  = np.array([c[0] for c in coeff_T]);   b_T  = np.array([c[1] for c in coeff_T])
        c_T  = np.array([c[2] for c in coeff_T])
        a_dT = np.array([c[0] for c in coeff_Tdot]); b_dT = np.array([c[1] for c in coeff_Tdot])
        c_dT = np.array([c[2] for c in coeff_Tdot]); d_dT = np.array([c[3] for c in coeff_Tdot])

        mTu  = (self.T_max     < np.inf) & (self.T_max     > 0)
        mTl  = (self.T_min     > -np.inf) & (self.T_min    < 0)
        mTdu = (self.T_dot_max < np.inf) & (self.T_dot_max > 0)
        mTdl = (self.T_dot_min > -np.inf) & (self.T_dot_min < 0)

        # ── FIX 1: anchor search scales ALL values including boundaries ───────
        x0_raw = np.maximum(np.array(self.v_init, dtype=float), 1e-9)
        x_anchor_raw = x0_raw.copy()

        def _raw_violation(x_raw):
            sp = self._build_spline(x_raw)
            sdot       = np.maximum(sp(s_check), 1e-9)
            dsdot_ds   = sp(s_check, 1)
            d2sdot_ds2 = sp(s_check, 2)
            sddot      = dsdot_ds * sdot
            sdddot     = sdot * (dsdot_ds**2 + sdot * d2sdot_ds2)
            tau  = a_T * sddot[:, None] + b_T * sdot[:, None]**2 + c_T
            tdot = (a_dT * sdddot[:, None] + b_dT * sdot[:, None] * sddot[:, None]
                    + c_dT * sdot[:, None]**3 + d_dT * sdot[:, None])
            vio = 0.0
            if np.any(mTu):  vio = max(vio, np.max(tau[:,  mTu]  / self.T_max[mTu]     - 1.0))
            if np.any(mTl):  vio = max(vio, np.max(tau[:,  mTl]  / self.T_min[mTl]     - 1.0))
            if np.any(mTdu): vio = max(vio, np.max(tdot[:, mTdu] / self.T_dot_max[mTdu] - 1.0))
            if np.any(mTdl): vio = max(vio, np.max(tdot[:, mTdl] / self.T_dot_min[mTdl] - 1.0))
            return vio

        for _ in range(120):
            if _raw_violation(x_anchor_raw) <= 0.0:
                break
            x_anchor_raw = np.maximum(x_anchor_raw * 0.6, 1e-9)

        # bc comes from the anchor's boundary values so spline stays consistent
        bc = (float(x_anchor_raw[0]), float(x_anchor_raw[-1]))
        x_anchor = x_anchor_raw.copy()

        if verbose:
            print(f"[FTM] Anchor: mean_sdot={np.mean(x_anchor):.3e}  "
                f"T(anchor)={_raw_violation(x_anchor):.2e}  bc={bc}")

        # ── _apply_bc uses the (possibly scaled) anchor boundary values ───────
        def _apply_bc(x):
            x = x.copy(); x[0] = bc[0]; x[-1] = bc[1]
            return np.maximum(x, 1e-9)

        # ── objective ─────────────────────────────────────────────────────────
        def objective(x):
            sp   = self._build_spline(_apply_bc(x))
            sdot = np.maximum(sp(_s_quad), 1e-9)
            return float(np.trapz(1.0 / sdot, _s_quad))

        # ── constraint violation T(x) ─────────────────────────────────────────
        def constraint_violation(x, stop_at=None):
            return _raw_violation(_apply_bc(x))   # reuse _raw_violation

        def is_acceptable(x, phi):
            return constraint_violation(x, stop_at=phi) <= phi

        # ── bisection ─────────────────────────────────────────────────────────
        def bisect_to_feasible(x_bad, x_hint, phi, max_steps=25):
            anchor = x_hint if is_acceptable(x_hint, phi) else x_anchor
            lo, hi, best = 0.0, 1.0, _apply_bc(anchor)
            for _ in range(max_steps):
                mid   = 0.5 * (lo + hi)
                x_mid = _apply_bc(x_bad + mid * (anchor - x_bad))
                if is_acceptable(x_mid, phi):
                    hi, best = mid, x_mid
                else:
                    lo = mid
            return _apply_bc(best)

        # ── initial phi ───────────────────────────────────────────────────────
        x0    = _apply_bc(x0_raw)
        v0vio = constraint_violation(x0)
        phi   = max(phi0 if phi0 is not None else 0.2, v0vio + 1e-3)

        if verbose:
            print(f"[FTM] T(x0)={v0vio:.3e}  phi0={phi:.3e}")

        # ── initialise simplex ────────────────────────────────────────────────
        verts = np.empty((n + 1, n))
        verts[0] = x0
        step = np.maximum(0.2 * x0, 0.05)
        for i in range(1, n + 1):
            xi = x0.copy()
            j  = (i - 1) % (n - 2) + 1
            xi[j] += step[j]
            verts[i] = _apply_bc(xi)

        f_vals = np.array([objective(v) for v in verts])
        T_vals = np.array([constraint_violation(v) for v in verts])

        for j in range(n + 1):
            if T_vals[j] > phi:
                verts[j]  = bisect_to_feasible(verts[j], x0, phi)
                f_vals[j] = objective(verts[j])
                T_vals[j] = constraint_violation(verts[j])

        best_idx = int(np.argmin(f_vals))
        best_x   = verts[best_idx].copy()
        best_f   = f_vals[best_idx]

        it = -1
        for it in range(max_iter):
            if (time.perf_counter() - t_start) > max_wall_time_s:
                if verbose:
                    print(f"[FTM] wall-time limit at iter {it}")
                break

            for j in range(n + 1):
                if not is_acceptable(verts[j], phi):
                    verts[j]  = bisect_to_feasible(verts[j], best_x, phi)
                    f_vals[j] = objective(verts[j])
                    T_vals[j] = constraint_violation(verts[j])

            order  = np.argsort(f_vals)
            verts  = verts[order]; f_vals = f_vals[order]; T_vals = T_vals[order]

            x_best  = verts[0].copy()
            x_worst = verts[-1]
            x_cen   = verts[:-1].mean(axis=0)

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

            x_ref = _apply_bc(x_cen + alpha * (x_cen - x_worst))
            if not is_acceptable(x_ref, phi):
                x_ref = bisect_to_feasible(x_ref, x_best, phi)
            f_ref = objective(x_ref); T_ref = constraint_violation(x_ref)

            if f_ref < f_vals[0]:
                x_exp = _apply_bc(x_cen + gamma * (x_cen - x_worst))
                if not is_acceptable(x_exp, phi):
                    x_exp = bisect_to_feasible(x_exp, x_best, phi)
                f_exp = objective(x_exp); T_exp = constraint_violation(x_exp)
                if f_exp < f_ref:
                    verts[-1], f_vals[-1], T_vals[-1] = x_exp, f_exp, T_exp
                else:
                    verts[-1], f_vals[-1], T_vals[-1] = x_ref, f_ref, T_ref
            elif f_ref < f_vals[-2]:
                verts[-1], f_vals[-1], T_vals[-1] = x_ref, f_ref, T_ref
            else:
                x_con = _apply_bc(x_cen + beta * (x_worst - x_cen))
                if not is_acceptable(x_con, phi):
                    x_con = bisect_to_feasible(x_con, x_best, phi)
                f_con = objective(x_con); T_con = constraint_violation(x_con)
                if f_con < f_vals[-1]:
                    verts[-1], f_vals[-1], T_vals[-1] = x_con, f_con, T_con
                else:
                    for j in range(1, n + 1):
                        verts[j]  = _apply_bc(x_best + beta * (verts[j] - x_best))
                        f_vals[j] = objective(verts[j])
                        T_vals[j] = constraint_violation(verts[j])

            for j in range(n + 1):
                if f_vals[j] < best_f and T_vals[j] <= phi:
                    best_f = f_vals[j]
                    best_x = verts[j].copy()

            # Eq. A5 — simplex spread (- sign), fresh centroid after any shrink
            x_cen_cur = verts[:-1].mean(axis=0)
            phi_new   = k_phi * float(np.sum(np.linalg.norm(verts - x_cen_cur, axis=1)))
            phi = min(phi, phi_new)
            # FIX 2: floor ensures best vertex always passes the gate
            phi = max(phi, T_vals[0] + 1e-6, tol)

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
