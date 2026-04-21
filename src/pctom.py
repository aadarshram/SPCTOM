"""
Implements an adapted version of path-constrained time-optimal motion planning (PCTOM).
"""

# Imports
import numpy as np

from src.tom import TOM

class PCTOM(TOM):
    """
    Ref: Shiller and Lu. Computation of Path Constrained Time Optimal Motions With Dynamic Singularities
    """

    def __init__(self, robot, path_param, s_values,
                 T_max=None, T_min=None, ds=None, boundary_conditions=None):
        super().__init__(
            path_param,
            robot,
            s_values,
            T_max=T_max,
            T_min=T_min,
            ds=ds,
            boundary_conditions=boundary_conditions,
        )

    def _trim_to(self, s_arr, sdot_arr, s_cross, sdot_cross):
        """Keep only elements with s <= s_cross, then append the crossing pt."""
        idx = np.searchsorted(s_arr, s_cross)
        s_out    = list(s_arr[:idx])    + [s_cross]
        sdot_out = list(sdot_arr[:idx]) + [sdot_cross]
        return s_out, sdot_out

    def _find_crossing(self, fwd_s, fwd_sdot, bwd_s, bwd_sdot):
        """
        Find crossing point of forward and backward trajectories.
        """
        # Interpolate both to a common s grid for easier crossing detection
        all_s = np.union1d(fwd_s, bwd_s)
        fwd_interp = np.interp(all_s, fwd_s, fwd_sdot)
        bwd_interp = np.interp(all_s, bwd_s, bwd_sdot)
        diff = fwd_interp - bwd_interp
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            # No crossing found; return closest point
            idx_closest = np.argmin(np.abs(diff))
            return all_s[idx_closest], fwd_interp[idx_closest]
        else:
            # Take first crossing
            # Interpolate to find more accurate crossing point and sdot at crossing
            idx = sign_changes[0]
            s1, s2 = all_s[idx], all_s[idx+1]
            d1, d2 = diff[idx], diff[idx+1]
            s_cross = s1 - d1 * (s2 - s1) / (d2 - d1)
            f1, f2 = fwd_interp[idx], fwd_interp[idx+1]
            sdot_cross = f1 + (s_cross - s1) * (f2 - f1) / (s2 - s1)
            return s_cross, sdot_cross
    
    # def _backward_from(self, s_start, sdot_start, fwd_s, fwd_sdot, max_steps=5000):
    #     """
    #     Integrate backward from (s_start, sdot_start) until crossing fwd traj.
    #     """
    #     ds = self.ds
    #     bwd_s    = [s_start]
    #     bwd_sdot = [sdot_start]
    #     bwd_sddot = [0.0]
    #     s, sdot  = s_start, sdot_start
    #     fwd_min_s = min(fwd_s)
    #     for _ in range(max_steps):
    #         if s <= fwd_min_s:
    #             break
    #         s_new, sdot_new = self._step(s, sdot, -1, ds)
    #         if s_new < fwd_min_s:
    #             break
    #         sdot_new = max(sdot_new, 1e-9)
    #         bwd_s.append(s_new)
    #         bwd_sdot.append(sdot_new)
    #         lo, _ = self._sddot_bounds_from_T(
    #             *self.get_T_coeffs(s_new),
    #             tau_min=self.T_min,
    #             tau_max=self.T_max,
    #             sdot=max(sdot_new, 1e-9),
    #         )
    #         bwd_sddot.append(lo)
    #         s, sdot = s_new, sdot_new

    #     return list(reversed(bwd_s)), list(reversed(bwd_sdot)), list(reversed(bwd_sddot))
    
    # def _find_next_switch(self, s_start):
    #     """
    #     Search forward from s_start along the VLC for the next critical/tangency point or discontinuity.
    #     Uses function _classify_limit_hit to check the type of limit hit at each point.
    #     Returns the s value of the next switching point.
    #     """
    #     s = s_start
    #     while s < self.s_values[-1] - self.ds:
    #         s += self.ds
    #         sw_type = self._classify_limit_hit(s, self._sdot_max_at(s))
    #         if sw_type in ['tangency', 'discontinuity', 'singular', 'critical']:
    #             return s, sw_type
    #     return self.s_values[-1], None # fallback to end if no switch found
    
    def _singular_acceleration(self, s, sdot):
        """
        Compute singular acceleration using Eq (17)/(18):
        sddot = sdot * d(sdot_max)/ds
        """
        s_lo = max(s - self.ds, self.s_values[0])
        s_hi = min(s + self.ds, self.s_values[-1])
        dvlc_ds = (self._sdot_max_at(s_hi) - self._sdot_max_at(s_lo)) / (s_hi - s_lo)
        return self._sdot_max_at(s) * dvlc_ds
    
    def _classify_limit_hit(self, s, sdot):
        """
        At (s, sdot≈Ṡ_max), determine whether this is a tangency,
        discontinuity, or singular (critical) point.

        Returns one of: 'tangency', 'discontinuity', 'singular', 'regular'
        """

        # Discontinuity: Large jump in VLC
        # Assume threshold of x for now, can be tuned later or made adaptive
        x = 0.3
        vlc_here  = self._sdot_max_at(s)
        vlc_after = self._sdot_max_at(min(s + self.ds, self.s_values[-1]))
        vlc_before= self._sdot_max_at(max(s - self.ds, self.s_values[0]))
        vlc_slope = (vlc_after - vlc_before) / (2 * self.ds)
        if abs(vlc_after - vlc_here) > x or abs(vlc_before - vlc_here) > x:
            return 'discontinuity'

        # Critical point: Constraint becomes independent of sddot , ie, a_i = 0 in Eq (6)
        is_critical = False
        a_prev, _, _ = self.get_T_coeffs(max(s-self.ds, self.s_values[0]))
        a_next, _, _ = self.get_T_coeffs(min(s+self.ds, self.s_values[-1]))
        # Check zero crossing of a_i
        for i in range(len(a_prev)):
             if a_prev[i] * a_next[i] <= 0:
                is_critical = True
                break
        if is_critical:
            # Check if singular
            lo, hi = self._sddot_bounds_from_T(*self.get_T_coeffs(s), tau_min=self.T_min, tau_max=self.T_max, sdot=max(sdot, 1e-9))
            traj_slope_accel = hi / max(sdot, 1e-9)
            traj_slope_decel = lo / max(sdot, 1e-9)
            # Eq(13) and Eq(14) violation for singularity
            if not (traj_slope_decel <= vlc_slope <= traj_slope_accel):
                return 'singular'
            else:
                return 'critical'

        # Tangency: phase-plane slope d(sdot)/ds = sddot/sdot matches VLC slope.
        # Use feasible accel bounds at the current limit point.
        y = 0.05
        lo, hi = self._sddot_bounds_from_T(
            *self.get_T_coeffs(s), tau_min=self.T_min, tau_max=self.T_max, sdot=max(sdot, 1e-9)
        )
        traj_slope_accel = hi / max(sdot, 1e-9)
        traj_slope_decel = lo / max(sdot, 1e-9)

        if min(abs(traj_slope_accel - vlc_slope), abs(traj_slope_decel - vlc_slope)) < y * (abs(vlc_slope) + 1e-6):
            return 'tangency'

        return 'regular' # fallback
    

    def _package(self, s_traj, sdot_traj, switching_s, switching_types):
        s_traj    = np.asarray(s_traj)
        sdot_traj = np.asarray(sdot_traj)
        sddot_traj = np.asarray(getattr(self, '_last_sddot_traj', np.zeros_like(sdot_traj)))
        sddot_min_traj = np.asarray(getattr(self, '_last_sddot_min_traj', np.full_like(sdot_traj, np.nan)))
        sddot_max_traj = np.asarray(getattr(self, '_last_sddot_max_traj', np.full_like(sdot_traj, np.nan)))
        # Motion time: sum per segment with
        #   dt = 2*Δs / (ṡ_i + ṡ_{i+1})
        # This matches v=ṡ^2 integration used by _step and avoids endpoint
        # overestimation from trapezoidal integration of 1/ṡ.
        sdot_time = np.maximum(sdot_traj.copy(), 1e-9)
        if s_traj.size > 1:
            ds_seg = np.diff(s_traj)
            denom = np.maximum(sdot_time[:-1] + sdot_time[1:], 1e-12)
            motion_time = float(np.sum(2.0 * ds_seg / denom))
        else:
            motion_time = 0.0

        return {
            's_traj':          s_traj,
            'sdot_traj':       sdot_traj,
            'sddot_traj':      sddot_traj,
            'sddot_min_traj':  sddot_min_traj,
            'sddot_max_traj':  sddot_max_traj,
            'motion_time':     motion_time,
            'switching_s':     np.array(switching_s),
            'switching_types': switching_types,
            'vlc_s':           self.s_values,
            'vlc_sdot':        self.sdot_max_values,
        }

    def _merge(self, fwd_s, fwd_sdot, bwd_s, bwd_sdot):
        """
        Merge forward and backward arcs, taking the minimum at each s.
        """
        all_s = np.union1d(fwd_s, bwd_s)
        # Outside each arc's support, set value to +inf so min() selects the
        # branch that is actually defined there (avoid endpoint extrapolation).
        fwd_interp = np.interp(all_s, fwd_s, fwd_sdot, left=np.inf, right=np.inf)
        bwd_interp = np.interp(all_s, bwd_s, bwd_sdot, left=np.inf, right=np.inf)
        sdot_merged = np.minimum(fwd_interp, bwd_interp) # PMP
        # Enforce velocity bounds- (0, VLC)
        vlc = np.array([self._sdot_max_at(s) for s in all_s])
        sdot_merged = np.minimum(sdot_merged, vlc)
        sdot_merged = np.maximum(sdot_merged, 1e-9)
        return all_s, sdot_merged
    
    # def _backward_final(self, sf, sdotf, fwd_s, fwd_sdot, max_steps=5000):
    #     """
    #     Backward integration from sf for the final decel arc.
    #     """
    #     ds = self.ds
    #     bwd_s    = [sf]
    #     bwd_sdot = [sdotf]
    #     bwd_sddot = [0.0]
    #     s, sdot  = sf, sdotf
    #     fwd_min_s = min(fwd_s) # Don't go back further than the start of the forward trajectory

    #     for _ in range(max_steps):
    #         if s <= fwd_min_s:
    #             break
    #         s_new, sdot_new = self._step(s, sdot, -1, ds)
    #         if s_new < fwd_min_s:
    #             break
    #         sdot_new = max(sdot_new, 1e-9)

    #         # Stop at the natural crossing with the forward branch.
    #         fwd_here = np.interp(s_new, fwd_s, fwd_sdot, left=np.nan, right=np.nan)
    #         if np.isfinite(fwd_here) and sdot_new >= fwd_here:
    #             bwd_s.append(s_new)
    #             bwd_sdot.append(sdot_new)
    #             lo, _ = self._sddot_bounds_from_T(
    #                 *self.get_T_coeffs(s_new),
    #                 tau_min=self.T_min,
    #                 tau_max=self.T_max,
    #                 sdot=max(sdot_new, 1e-9),
    #             )
    #             bwd_sddot.append(lo)
    #             break

    #         bwd_s.append(s_new)
    #         bwd_sdot.append(sdot_new)
    #         lo, _ = self._sddot_bounds_from_T(
    #             *self.get_T_coeffs(s_new),
    #             tau_min=self.T_min,
    #             tau_max=self.T_max,
    #             sdot=max(sdot_new, 1e-9),
    #         )
    #         bwd_sddot.append(lo)
    #         s, sdot = s_new, sdot_new
    #     return list(reversed(bwd_s)), list(reversed(bwd_sdot)), list(reversed(bwd_sddot))
    
    def _step(self, s, sdot, direction, ds, sddot=None):
        """
        Integrate one step in the (s, ṡ) plane.
        If sddot is given, integrates directly else uses min/max acc based on direction: +1 (forward, max accel) or -1 (backward, max decel).
        Returns (s_new, sdot_new)

        """
        def sddot_at(s, sdot):
            sddot_min, sddot_max = self._sddot_bounds_from_T(*self.get_T_coeffs(s), tau_min=self.T_min, tau_max=self.T_max, sdot=sdot)
            return sddot_max if direction == +1 else sddot_min

        # Midpoint RK2 in v = sdot^2 space:
        #   dv/ds = 2*sddot
        # This avoids singular behavior at sdot -> 0.
        if sddot is None:
            sddot1 = float(sddot_at(s, sdot))
        elif sddot == 'singular':
            sddot1 = float(self._singular_acceleration(s, sdot))
        else:
            sddot1 = float(sddot)
        s_mid  = s    + 0.5 * ds * direction
        v = max(sdot, 1e-9) ** 2
        v_mid = max(v + 0.5 * ds * direction * 2.0 * sddot1, 1e-18)
        sdot_mid = np.sqrt(v_mid)

        if sddot is None:
            sddot2 = float(sddot_at(s_mid, sdot_mid))
        elif sddot == 'singular':
            sddot2 = float(self._singular_acceleration(s_mid, sdot_mid))
        else:
            sddot2 = float(sddot)

        v_new = max(v + ds * direction * (sddot1 + sddot2), 1e-18)
        s_new    = s + ds * direction
        sdot_new = np.sqrt(v_new)
        return s_new, sdot_new

    # def solve(self, verbose=False):
    #     """
    #     Compute the time-optimal trajectory via PCTOM method

    #     Returns:
    #     Dict:
    #         {
    #         's_traj'          – s values of the optimal trajectory
    #         'sdot_traj'       – ṡ values
    #         'motion_time'     – total time  ∫ ds/ṡ
    #         'switching_s'     – s values of all switching points
    #         'switching_types' – corresponding type strings
    #         'vlc_s'           – s grid for velocity limit curve
    #         'vlc_sdot'        – Ṡ_max values on that grid
    #         }
    #     """
        
    #     # Initialize
    #     s0 = self.s_values[0]
    #     sf = self.s_values[-1]
    #     sdot0 = self.sdot0
    #     sdotf = self.sdotf
    #     ds = self.ds
    #     switching_s     = [] # Store s values of switching points (critical, tangency, discontinuity, singular)
    #     switching_types = [] # Store corresponding types (critical, tangency, discontinuity, singular)

    #     # Step 1: Forward integration from s0 with max accel until we hit VLC or reach sf.
    #     # Bootstrap near-zero initial speed to avoid phase-plane singularity at sdot≈0.
    #     if sdot0 < 1e-8:
    #         lo0, hi0 = self._sddot_bounds_from_T(
    #             *self.get_T_coeffs(s0),
    #             tau_min=self.T_min,
    #             tau_max=self.T_max,
    #             sdot=max(sdot0, 1e-9),
    #         )
    #         if np.isfinite(hi0) and hi0 > 0.0:
    #             sdot0 = np.sqrt(max(0.0, 2.0 * hi0 * ds))
    #         sdot0 = min(sdot0, self._sdot_max_at(s0) * 0.9999)

    #     fwd_s    = [s0]
    #     fwd_sdot = [sdot0]
    #     fwd_sddot = [0.0]
    #     s, sdot = s0, sdot0
    #     singular_mode = False
    #     # Steps 2–5: keep handling limit hits until we reach sf.
    #     while s < sf - ds:
    #         if singular_mode:
    #             # Singular continuation: keep using singular acceleration until the
    #             # trajectory reaches the next VLC hit, then hand control back to Step 2.
    #             s_new, sdot_new = self._step(
    #                 s,
    #                 sdot,
    #                 +1,
    #                 ds,
    #                 sddot='singular',
    #             )
    #             vlc_new = self._sdot_max_at(s_new)
    #             if sdot_new < vlc_new:
    #                 s, sdot = s_new, sdot_new
    #                 fwd_s.append(s)
    #                 fwd_sdot.append(sdot)
    #                 fwd_sddot.append(self._singular_acceleration(s, sdot))
                    
    #                 continue

    #             singular_mode = False
    #             s_hit = s_new
    #             sdot_hit = max(vlc_new * 0.9999, 1e-9)
    #             sw_type = self._classify_limit_hit(s_hit, sdot_hit)
    #             if verbose:
    #                 print(f"[PCTOM] Singular arc hit VLC at s={s_hit:.4f}, type={sw_type}")
    #             switching_s.append(s_hit)
    #             switching_types.append(sw_type)
    #             s, sdot = s_hit, sdot_hit
    #             fwd_s.append(s)
    #             fwd_sdot.append(sdot)
    #             fwd_sddot.append(self._singular_acceleration(s, sdot))
    #             continue

    #         s_new, sdot_new = self._step(s, sdot, +1, ds)
    #         vlc_new = self._sdot_max_at(s_new)
    #         if sdot_new < vlc_new:
    #             s, sdot = s_new, sdot_new
    #             fwd_s.append(s)
    #             fwd_sdot.append(sdot)
    #             lo, hi = self._sddot_bounds_from_T(
    #                 *self.get_T_coeffs(s),
    #                 tau_min=self.T_min,
    #                 tau_max=self.T_max,
    #                 sdot=max(sdot, 1e-9),
    #             )
    #             fwd_sddot.append(hi)
    #             continue

    #         # Step 2: Hit VLC and classify the point type (singular / tangency / discontinuity / critical / regular).
    #         s_hit = s_new
    #         sdot_hit = max(vlc_new * 0.9999, 1e-9)
    #         sw_type = self._classify_limit_hit(s_hit, sdot_hit)
    #         if verbose:
    #             print(f"[PCTOM] Limit hit at s={s_hit:.4f}, type={sw_type}")

    #         if sw_type == 'singular':
    #             # Step 5 (singular branch): continue forward using singular acceleration (Eq. 17/18),
    #             # then remain in singular mode until the next VLC hit.
    #             singular_mode = True
    #             s = s_hit
    #             sdot = sdot_hit
    #             fwd_s.append(s)
    #             fwd_sdot.append(sdot)
    #             fwd_sddot.append(self._singular_acceleration(s, sdot))
    #             switching_s.append(s_hit)
    #             switching_types.append('singular')
    #             continue

    #         # Step 3: For non-singular hit, search forward for next switch candidate on/near VLC.
    #         s_tan, sw2 = self._find_next_switch(s_hit)
    #         if verbose:
    #             print(f"[PCTOM] Next switch at s={s_tan:.4f}, type={sw2}")
    #         if sw2 is not None:
    #             switching_s.append(s_tan)
    #             switching_types.append(sw2)

    #         # Step 4: Backward integrate from that candidate until crossing previous forward arc.
    #         bwd_s, bwd_sdot, bwd_sddot = self._backward_from(
    #             s_tan,
    #             self._sdot_max_at(s_tan) * 0.9999,
    #             fwd_s,
    #             fwd_sdot,
    #         )
    #         s_cr, sdot_cr = self._find_crossing(fwd_s, fwd_sdot, bwd_s, bwd_sdot)
    #         if verbose:
    #             print(f"[PCTOM] Switch at s_cr={s_cr:.4f}, sdot={sdot_cr:.4f}")
    #         switching_s.append(s_cr)
    #         switching_types.append('switch')

    #         # Step 5 (normal branch): trim at crossing and stitch backward segment,
    #         # then continue forward and loop back to Step 2 at the next VLC hit.
    #         idx = np.searchsorted(fwd_s, s_cr)
    #         fwd_s, fwd_sdot = self._trim_to(fwd_s, fwd_sdot, s_cr, sdot_cr)
    #         fwd_sddot = list(fwd_sddot[:idx]) + [0.0]
    #         for sb, sdb, sddb in zip(bwd_s, bwd_sdot, bwd_sddot):
    #             if sb > s_cr + 1e-12:
    #                 fwd_s.append(sb)
    #                 fwd_sdot.append(max(sdb, 1e-9))
    #                 fwd_sddot.append(sddb)

    #         s, sdot = fwd_s[-1], fwd_sdot[-1]

    #     # Step 6: Final backward integration from sf to satisfy terminal speed boundary.
    #     bwd_s, bwd_sdot, bwd_sddot = self._backward_final(sf, sdotf, fwd_s, fwd_sdot)
    #     s_traj, sdot_traj = self._merge(fwd_s, fwd_sdot, bwd_s, bwd_sdot)

    #     # Merge sddot from whichever branch is active in the merged sdot profile.
    #     all_s = np.asarray(s_traj)
    #     fwd_sdot_interp = np.interp(all_s, fwd_s, fwd_sdot, left=np.inf, right=np.inf)
    #     bwd_sdot_interp = np.interp(all_s, bwd_s, bwd_sdot, left=np.inf, right=np.inf)
    #     fwd_sddot_interp = np.interp(all_s, fwd_s, fwd_sddot, left=np.nan, right=np.nan)
    #     bwd_sddot_interp = np.interp(all_s, bwd_s, bwd_sddot, left=np.nan, right=np.nan)
    #     use_fwd = fwd_sdot_interp <= bwd_sdot_interp
    #     sddot_traj = np.where(use_fwd, fwd_sddot_interp, bwd_sddot_interp)
    #     self._last_sddot_traj = np.nan_to_num(sddot_traj, nan=0.0)

    #     # Record acceleration bounds along the final merged trajectory.
    #     sddot_min_traj = np.zeros_like(all_s, dtype=float)
    #     sddot_max_traj = np.zeros_like(all_s, dtype=float)
    #     for i, (si, sdi) in enumerate(zip(all_s, sdot_traj)):
    #         lo, hi = self._sddot_bounds_from_T(
    #             *self.get_T_coeffs(float(si)),
    #             tau_min=self.T_min,
    #             tau_max=self.T_max,
    #             sdot=float(max(sdi, 1e-9)),
    #         )
    #         sddot_min_traj[i] = lo
    #         sddot_max_traj[i] = hi
    #     self._last_sddot_min_traj = sddot_min_traj
    #     self._last_sddot_max_traj = sddot_max_traj

    #     return self._package(s_traj, sdot_traj, switching_s, switching_types)
# ------------------------------------------------------------------ #
    #  Helper: central-difference VLC slope                               #
    # ------------------------------------------------------------------ #
    def _vlc_slope(self, s):
        s_lo = max(s - self.ds, self.s_values[0])
        s_hi = min(s + self.ds, self.s_values[-1])
        return (self._sdot_max_at(s_hi) - self._sdot_max_at(s_lo)) / (s_hi - s_lo)

    # ------------------------------------------------------------------ #
    #  Helper: paper Eqs (13)/(14) singular check                         #
    # ------------------------------------------------------------------ #
    def _is_singular_point(self, s, sdot):
        """
        Returns True if (s, sdot) is a singular/dynamic-singularity point.
        Paper condition: the VLC slope falls OUTSIDE the feasible phase-plane
        slope band [lo/sdot, hi/sdot]  (Eqs 13 & 14 are violated).
        """
        lo, hi = self._sddot_bounds_from_T(
            *self.get_T_coeffs(s),
            tau_min=self.T_min,
            tau_max=self.T_max,
            sdot=max(sdot, 1e-9),
        )
        vlc_sl = self._vlc_slope(s)
        accel_slope = hi / max(sdot, 1e-9)
        decel_slope = lo / max(sdot, 1e-9)
        # Singular when VLC slope is NOT bracketed by feasible slopes
        return not (decel_slope <= vlc_sl <= accel_slope)

    # ------------------------------------------------------------------ #
    #  Helper: refined crossing point between backward and forward arcs   #
    # ------------------------------------------------------------------ #
    def _crossing_point(self, s_prev, sdot_prev, s_new, sdot_new,
                    fwd_s, fwd_sdot, n_bisect=8):
        """
        Refine crossing by bisection between (s_prev, sdot_prev) 
        and (s_new, sdot_new).
        """
        sl, sr = s_prev, s_new
        vl, vr = sdot_prev, sdot_new

        for _ in range(n_bisect):
            sm = 0.5 * (sl + sr)
            vm = 0.5 * (vl + vr)
            fwd_m = float(np.interp(sm, fwd_s, fwd_sdot,
                                    left=np.nan, right=np.nan))
            if not np.isfinite(fwd_m):
                break
            if vm <= fwd_m:
                sl, vl = sm, vm
            else:
                sr, vr = sm, vm
        s_cross    = 0.5 * (sl + sr)
        sdot_cross = float(np.interp(s_cross, fwd_s, fwd_sdot,
                                    left=np.nan, right=np.nan))
        if not np.isfinite(sdot_cross):
            sdot_cross = 0.5 * (vl + vr)
        return s_cross, sdot_cross

    # ------------------------------------------------------------------ #
    #  Rewritten _backward_from: stops as soon as crossing is detected    #
    # ------------------------------------------------------------------ #
    def _backward_from(self, s_start, sdot_start, fwd_s, fwd_sdot,
                       max_steps=5000):
        """
        Integrate backward from (s_start, sdot_start) with max feasible
        deceleration until the arc crosses the forward trajectory.
        Crossing is detected *inside* the loop and refined linearly.
        """
        ds = self.ds
        bwd_s    = [s_start]
        bwd_sdot = [sdot_start]
        s, sdot  = s_start, sdot_start
        fwd_min_s = float(min(fwd_s))

        for _ in range(max_steps):
            if s <= fwd_min_s + ds:
                break

            s_new, sdot_new = self._step(s, sdot, -1, ds)
            sdot_new = max(sdot_new, 1e-9)

            if s_new < fwd_min_s:
                break

            # --- crossing check BEFORE appending ---
            fwd_here = float(np.interp(s_new, fwd_s, fwd_sdot,
                                       left=np.nan, right=np.nan))
            if np.isfinite(fwd_here) and sdot_new <= fwd_here:
                s_cr, sdot_cr = self._crossing_point(
                    s, sdot, s_new, sdot_new, fwd_s, fwd_sdot)
                bwd_s.append(s_cr)
                bwd_sdot.append(sdot_cr)
                break

            bwd_s.append(s_new)
            bwd_sdot.append(sdot_new)
            s, sdot = s_new, sdot_new

        return list(reversed(bwd_s)), list(reversed(bwd_sdot))

    # ------------------------------------------------------------------ #
    #  Rewritten _find_next_switch: separate critical / tangency checks   #
    # ------------------------------------------------------------------ #
    def _find_next_switch(self, s_start):
        """
        Detect nearest critical or tangency point using sign-change logic.
        """
        s = s_start

        # For discontinuity detection
        if self._is_discontinuity(s):
            return s, 'discontinuity'

        # For tangency detection
        lo0, hi0 = self._sddot_bounds_from_T(
            *self.get_T_coeffs(s_start),
            tau_min=self.T_min,
            tau_max=self.T_max,
            sdot=max(self._sdot_max_at(s_start) * 0.9999, 1e-9),
        )
        vlc_sl0 = self._vlc_slope(s_start)
        accel_sl0 = hi0 / max(self._sdot_max_at(s_start) * 0.9999, 1e-9)
        decel_sl0 = lo0 / max(self._sdot_max_at(s_start) * 0.9999, 1e-9)
        prev_diff = accel_sl0 - vlc_sl0
        prev_decel_diff = decel_sl0 - vlc_sl0

        # For critical detection
        a_prev, _, _ = self.get_T_coeffs(s)

        while s < self.s_values[-1] - self.ds:
            s += self.ds
            sdot = self._sdot_max_at(s)

            # -------- Critical detection (FIXED in next section) --------
            a_curr, _, _ = self.get_T_coeffs(s)
            for i, (ap, ac) in enumerate(zip(a_prev, a_curr)):
                if ap * ac <= 0:
                    # Only accept if this constraint is ACTIVE
                    if self._is_active_constraint(i, s, sdot):
                        return s, 'critical'
            a_prev = a_curr

            # -------- Tangency detection (SIGN CHANGE) --------
            lo, hi = self._sddot_bounds_from_T(
                *self.get_T_coeffs(s),
                tau_min=self.T_min,
                tau_max=self.T_max,
                sdot=max(sdot, 1e-9),
            )

            vlc_sl = self._vlc_slope(s)
            accel_slope = hi / max(sdot, 1e-9)
            decel_slope = lo / max(sdot, 1e-9)

            diff = accel_slope - vlc_sl
            decel_diff = decel_slope - vlc_sl

            if (prev_diff * diff <= 0):
                return s, 'tangency'

            prev_diff = diff
            prev_decel_diff = decel_diff

        return self.s_values[-1], None
    
    def _is_active_constraint(self, idx, s, sdot, tol=1e-4):
        """
        Check if constraint i is ACTIVE (i.e., defines the bound).
        """
        a, b, c = self.get_T_coeffs(s)

        lo, hi = self._sddot_bounds_from_T(
            a, b, c,
            tau_min=self.T_min,
            tau_max=self.T_max,
            sdot=max(sdot, 1e-9),
        )

        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False

        # Evaluate this joint's torque at both accel bounds.
        tau_lo = a[idx] * lo + b[idx] * sdot**2 + c[idx]
        tau_hi = a[idx] * hi + b[idx] * sdot**2 + c[idx]

        if abs(tau_lo - self.T_max[idx]) < tol or abs(tau_lo - self.T_min[idx]) < tol:
            return True
        if abs(tau_hi - self.T_max[idx]) < tol or abs(tau_hi - self.T_min[idx]) < tol:
            return True

        return False
    # ------------------------------------------------------------------ #
    #  Rewritten _backward_final: same inline crossing logic              #
    # ------------------------------------------------------------------ #
    def _backward_final(self, sf, sdotf, fwd_s, fwd_sdot, max_steps=5000):
        """
        Backward integration from (sf, sdotf) for the terminal decel arc.
        Stops when the arc crosses the forward trajectory.
        """
        ds = self.ds
        bwd_s    = [sf]
        bwd_sdot = [sdotf]
        s, sdot  = sf, sdotf
        fwd_min_s = float(min(fwd_s))

        for _ in range(max_steps):
            if s <= fwd_min_s + ds:
                break

            s_new, sdot_new = self._step(s, sdot, -1, ds)
            sdot_new = max(sdot_new, 1e-9)

            if s_new < fwd_min_s:
                break

            # --- inline crossing check ---
            fwd_here = float(np.interp(s_new, fwd_s, fwd_sdot,
                                       left=np.nan, right=np.nan))
            if np.isfinite(fwd_here) and sdot_new >= fwd_here:
                s_cr, sdot_cr = self._crossing_point(
                    s, sdot, s_new, sdot_new, fwd_s, fwd_sdot)
                bwd_s.append(s_cr)
                bwd_sdot.append(sdot_cr)
                break

            bwd_s.append(s_new)
            bwd_sdot.append(sdot_new)
            s, sdot = s_new, sdot_new

        return list(reversed(bwd_s)), list(reversed(bwd_sdot))
    
    def _is_discontinuity(self, s, threshold=0.2):
        vlc_here = self._sdot_max_at(s)
        vlc_next = self._sdot_max_at(min(s + self.ds, self.s_values[-1]))

        return abs(vlc_next - vlc_here) > threshold * max(1.0, vlc_here)
    # ------------------------------------------------------------------ #
    #  SOLVE                                                               #
    # ------------------------------------------------------------------ #
    def solve(self, verbose=False):
        """
        Compute the time-optimal trajectory via the Shiller-Lu PCTOM method.

        Algorithm (paper Section 3):
          1. Forward max-accel until VLC hit at S_h.
          2. At S_h: singular check (Eqs 13/14).
             - Singular  → set S_t = S_h, go to Step 5.
             - Otherwise → go to Step 3.
          3. Search forward for nearest critical/tangency point S_t ≥ S_h.
          4. Backward from S_t with max feasible decel until crossing
             previous forward arc at S_cr ≤ S_h.
             Trajectory switches accel→decel at S_cr.
          5. Forward from S_t with max feasible accel (singular Eq 17/18)
             until VLC hit again → go to Step 2.
          6. Backward from final point until crossing forward arc.

        Returns
        -------
        dict with keys:
            s_traj, sdot_traj, sddot_traj,
            sddot_min_traj, sddot_max_traj,
            motion_time, switching_s, switching_types,
            vlc_s, vlc_sdot
        """
        s0    = self.s_values[0]
        sf    = self.s_values[-1]
        sdot0 = self.sdot0
        sdotf = self.sdotf
        ds    = self.ds

        switching_s     = []
        switching_types = []

        # ----------------------------------------------------------------
        # Bootstrap near-zero initial speed to avoid ṡ=0 singularity
        # ----------------------------------------------------------------
        if sdot0 < 1e-8:
            lo0, hi0 = self._sddot_bounds_from_T(
                *self.get_T_coeffs(s0),
                tau_min=self.T_min,
                tau_max=self.T_max,
                sdot=1e-9,
            )
            if np.isfinite(hi0) and hi0 > 0.0:
                sdot0 = np.sqrt(max(0.0, 2.0 * hi0 * ds))
            sdot0 = min(sdot0, self._sdot_max_at(s0) * 0.9999)

        # ----------------------------------------------------------------
        # Step 1: forward arc initialisation
        # ----------------------------------------------------------------
        fwd_s     = [s0]
        fwd_sdot  = [sdot0]
        fwd_sddot = [0.0]
        s, sdot   = s0, sdot0
        singular_mode = False

        # ================================================================
        # Main loop  — Steps 1–5
        # ================================================================
        while s < sf - ds:

            # ------------------------------------------------------------
            # Step 5 (singular continuation)
            # Integrate forward along singular arc (Eq 17/18), clamped to
            # the VLC, until the arc leaves the VLC again.
            # Then hand back to Step 2 at the new VLC hit.
            # ------------------------------------------------------------
            if singular_mode:
                s_new, sdot_new = self._step(s, sdot, +1, ds, sddot='singular')
                vlc_new  = self._sdot_max_at(s_new)
                # Clamp: singular arc rides the VLC
                sdot_new = min(sdot_new, vlc_new)
                sdot_new = max(sdot_new, 1e-9)

                fwd_s.append(s_new)
                fwd_sdot.append(sdot_new)
                fwd_sddot.append(self._singular_acceleration(s_new, sdot_new))
                s, sdot = s_new, sdot_new

                # Check whether we have left the VLC (next step would go below it)
                if not self._is_singular_point(s, sdot):
                    singular_mode = False
                    if verbose:
                        print(f"[PCTOM] Singular arc ended at s={s:.4f}")
                continue

            # ------------------------------------------------------------
            # Step 1: one forward max-accel step
            # ------------------------------------------------------------
            s_new, sdot_new = self._step(s, sdot, +1, ds)
            vlc_new = self._sdot_max_at(s_new)

            if sdot_new < vlc_new:
                # Still below VLC — keep going
                s, sdot = s_new, sdot_new
                fwd_s.append(s)
                fwd_sdot.append(sdot)
                lo, hi = self._sddot_bounds_from_T(
                    *self.get_T_coeffs(s),
                    tau_min=self.T_min,
                    tau_max=self.T_max,
                    sdot=max(sdot, 1e-9),
                )
                fwd_sddot.append(hi)
                continue

            # ------------------------------------------------------------
            # Trajectory hit the VLC at s_hit
            # ------------------------------------------------------------
            s_hit    = s_new
            sdot_hit = max(vlc_new * 0.9999, 1e-9)

            if verbose:
                print(f"[PCTOM] VLC hit at s={s_hit:.4f}, sdot={sdot_hit:.4f}")

            # ------------------------------------------------------------
            # Step 2: singular check using Eqs (13)/(14)
            # ------------------------------------------------------------
            if self._is_singular_point(s_hit, sdot_hit):
                # --- Singular point: S_t = S_h, go to Step 5 ---
                if verbose:
                    print(f"[PCTOM] Singular point at s={s_hit:.4f}")
                switching_s.append(s_hit)
                switching_types.append('singular')
                fwd_s.append(s_hit)
                fwd_sdot.append(sdot_hit)
                fwd_sddot.append(self._singular_acceleration(s_hit, sdot_hit))
                s, sdot       = s_hit, sdot_hit
                singular_mode = True
                continue

            # --- Not singular: go to Step 3 ---

            # ------------------------------------------------------------
            # Step 3: search forward for nearest critical / tangency point
            # ------------------------------------------------------------
            s_t, sw_type = self._find_next_switch(s_hit)
            sdot_t       = self._sdot_max_at(s_t) * 0.9999

            if verbose:
                print(f"[PCTOM] Next switch S_t={s_t:.4f}, type={sw_type}")

            if sw_type is not None:
                switching_s.append(s_t)
                switching_types.append(sw_type)

            # ------------------------------------------------------------
            # Step 4: backward from S_t until crossing previous fwd arc
            # ------------------------------------------------------------
            bwd_s, bwd_sdot = self._backward_from(
                s_t, sdot_t, fwd_s, fwd_sdot)

            if len(bwd_s) < 2:
                # Degenerate backward arc — skip and continue forward
                if verbose:
                    print(f"[PCTOM] Degenerate backward arc at s={s_t:.4f}, skipping")
                s, sdot = s_hit, sdot_hit
                fwd_s.append(s)
                fwd_sdot.append(sdot)
                fwd_sddot.append(0.0)
                continue

            # Crossing point S_cr is at the START of the (reversed) backward arc
            s_cr    = float(bwd_s[0])
            sdot_cr = float(bwd_sdot[0])

            if verbose:
                print(f"[PCTOM] Switch point S_cr={s_cr:.4f}, sdot={sdot_cr:.4f}")

            switching_s.append(s_cr)
            switching_types.append('switch')

            # ------------------------------------------------------------
            # Stitch: trim forward arc to S_cr, then append backward arc
            # (accel → decel switch at S_cr)
            # ------------------------------------------------------------
            idx = int(np.searchsorted(np.asarray(fwd_s), s_cr))
            fwd_s     = list(fwd_s[:idx])     + [s_cr]
            fwd_sdot  = list(fwd_sdot[:idx])  + [sdot_cr]
            fwd_sddot = list(fwd_sddot[:idx]) + [0.0]

            for sb, sdb in zip(bwd_s[1:], bwd_sdot[1:]):
                fwd_s.append(float(sb))
                fwd_sdot.append(max(float(sdb), 1e-9))
                # Recompute sddot for each stitched backward point
                lo_b, _ = self._sddot_bounds_from_T(
                    *self.get_T_coeffs(float(sb)),
                    tau_min=self.T_min,
                    tau_max=self.T_max,
                    sdot=max(float(sdb), 1e-9),
                )
                fwd_sddot.append(lo_b)
            
            # Clamp to VLC to avoid numerical issues
            for k in range(len(fwd_sdot)):
                fwd_sdot[k] = min(fwd_sdot[k], self._sdot_max_at(fwd_s[k]))
            fwd_sdot[-1] = max(fwd_sdot[-1], 1e-9)

            s, sdot = fwd_s[-1], fwd_sdot[-1]

            # ------------------------------------------------------------
            # Step 5 (non-singular branch): continue forward from S_t
            # The loop will naturally pick up from the new (s, sdot) which
            # is now at or near S_t and will hit the VLC again, returning
            # to Step 2. No special action needed here.
            # ------------------------------------------------------------

        # ================================================================
        # Step 6: backward from terminal point (sf, sdotf)
        # ================================================================
        if sdotf < 1e-8:
            lo_f, hi_f = self._sddot_bounds_from_T(
                *self.get_T_coeffs(sf),
                tau_min=self.T_min,
                tau_max=self.T_max,
                sdot=1e-9,
            )
            if np.isfinite(hi_f) and hi_f > 0.0:
                sdotf = np.sqrt(max(0.0, 2.0 * hi_f * ds))
            sdotf = min(sdotf, self._sdot_max_at(sf) * 0.9999)

        bwd_s, bwd_sdot = self._backward_final(sf, sdotf, fwd_s, fwd_sdot)

        # Record final crossing switch from terminal backward arc, if present.
        # In simple bang-bang cases this is the primary accel->decel switch.
        if len(bwd_s) > 0 and len(fwd_s) > 1:
            s_cr_final = float(bwd_s[0])
            sdot_cr_final = float(bwd_sdot[0])
            fwd_at_cr = float(np.interp(s_cr_final, fwd_s, fwd_sdot,
                                        left=np.nan, right=np.nan))
            if np.isfinite(fwd_at_cr):
                tol = max(1e-4, 1e-3 * max(1.0, abs(fwd_at_cr)))
                if abs(sdot_cr_final - fwd_at_cr) <= tol:
                    switching_s.append(s_cr_final)
                    switching_types.append('switch')
                    if verbose:
                        print(f"[PCTOM] Final switch S_cr={s_cr_final:.4f}, sdot={sdot_cr_final:.4f}")

        # ================================================================
        # Merge forward and final backward arcs (PMP: take minimum)
        # ================================================================
        all_s_set = sorted(set(list(fwd_s) + list(bwd_s)))
        all_s     = np.array(all_s_set)

        fwd_interp = np.interp(all_s, fwd_s, fwd_sdot,
                               left=np.inf, right=np.inf)
        bwd_interp = np.interp(all_s, bwd_s, bwd_sdot,
                               left=np.inf, right=np.inf)

        sdot_merged = np.minimum(fwd_interp, bwd_interp)

        # Enforce VLC and positivity
        vlc_arr     = np.array([self._sdot_max_at(si) for si in all_s])
        sdot_merged = np.minimum(sdot_merged, vlc_arr)
        sdot_merged = np.maximum(sdot_merged, 1e-9)

        # ================================================================
        # Build sddot along merged trajectory (consistent with final sdot)
        # ================================================================
        # Use v = sdot^2 so that sddot = 0.5 * dv/ds.
        v = np.maximum(sdot_merged, 1e-9) ** 2
        dv_ds = np.gradient(v, all_s, edge_order=1)
        sddot_merged = 0.5 * dv_ds

        # ================================================================
        # Acceleration bounds along merged trajectory
        # ================================================================
        sddot_min_traj = np.zeros_like(all_s)
        sddot_max_traj = np.zeros_like(all_s)
        for i, (si, sdi) in enumerate(zip(all_s, sdot_merged)):
            lo, hi = self._sddot_bounds_from_T(
                *self.get_T_coeffs(float(si)),
                tau_min=self.T_min,
                tau_max=self.T_max,
                sdot=float(max(sdi, 1e-9)),
            )
            sddot_min_traj[i] = lo
            sddot_max_traj[i] = hi

        self._last_sddot_traj = sddot_merged
        self._last_sddot_min_traj = sddot_min_traj
        self._last_sddot_max_traj = sddot_max_traj

        # ================================================================
        # Deduplicate switching points (within 1 ds tolerance)
        # ================================================================
        if len(switching_s) > 0:
            sw_arr   = np.array(switching_s)
            sw_types = switching_types
            order    = np.argsort(sw_arr)
            sw_arr   = sw_arr[order]
            sw_types = [sw_types[i] for i in order]
            keep     = np.concatenate(
                ([True], np.diff(sw_arr) > ds * 1.0))
            switching_s     = sw_arr[keep].tolist()
            switching_types = [sw_types[i]
                               for i, k in enumerate(keep) if k]

        return self._package(
            all_s, sdot_merged, switching_s, switching_types)

    def plot_solution(self, solution, ax=None):
        """
        Plot optimal sdot(s) with VLC and switching points.

        Parameters
        ----------
        solution : dict
            Output dictionary from solve().
        ax : matplotlib axis, optional
            Existing axis to draw on.

        Returns
        -------
        (fig, ax)
        """
        import matplotlib.pyplot as plt

        s = np.asarray(solution['s_traj'])
        sdot = np.asarray(solution['sdot_traj'])
        vlc_s = np.asarray(solution.get('vlc_s', self.s_values))
        vlc_sdot = np.asarray(solution.get('vlc_sdot', self.sdot_max_values))
        sw_s = np.asarray(solution.get('switching_s', np.array([])))
        sw_types = list(solution.get('switching_types', []))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.plot(s, sdot, linewidth=2.0, label='Optimal $\\dot{s}(s)$')
        ax.plot(vlc_s, vlc_sdot, linestyle=':', linewidth=2.0, label='VLC')

        # Mark switching points with per-type legend entries.
        marker_map = {
            'switch': ('o', 'Switch'),
            'singular': ('s', 'Singular'),
            'critical': ('^', 'Critical'),
            'tangency': ('D', 'Tangency'),
            'discontinuity': ('P', 'Discontinuity'),
            'regular': ('x', 'Regular-hit'),
        }
        shown = set()
        for i, s_i in enumerate(sw_s):
            sw_t = sw_types[i] if i < len(sw_types) else 'switch'
            marker, label = marker_map.get(sw_t, ('x', sw_t))
            lbl = label if label not in shown else None
            shown.add(label)
            sdot_i = float(np.interp(s_i, s, sdot))
            ax.scatter([s_i], [sdot_i], marker=marker, s=55, label=lbl, zorder=5)

        ax.set_xlabel('s')
        ax.set_ylabel('$\\dot{s}$')
        ax.set_title('PCTOM Solution and VLC')
        ax.grid(True, alpha=0.35)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig, ax

    def plot_optimal_torques(self, solution, ax=None):
        """
        Plot optimal actuator torques tau_i(s) for the solved trajectory.

        Parameters
        ----------
        solution : dict
            Output dictionary from solve().
        ax : matplotlib axis, optional
            Existing axis to draw on.

        Returns
        -------
        (fig, ax, tau)
            tau shape is (len(s_traj), n_joints).
        """
        import matplotlib.pyplot as plt

        s = np.asarray(solution['s_traj'])
        sdot = np.asarray(solution['sdot_traj'])
        sddot = np.asarray(solution.get('sddot_traj'))
        sddot_min = np.asarray(solution.get('sddot_min_traj'))
        sddot_max = np.asarray(solution.get('sddot_max_traj'))
        sw_s = np.asarray(solution.get('switching_s', np.array([])))
        sw_types = list(solution.get('switching_types', []))
        if sddot is None or sddot.shape != sdot.shape:
            sddot = np.zeros_like(sdot)
        if sddot_min is None or sddot_min.shape != sdot.shape:
            sddot_min = np.full_like(sdot, np.nan)
        if sddot_max is None or sddot_max.shape != sdot.shape:
            sddot_max = np.full_like(sdot, np.nan)

        tau = np.vstack([
            self.compute_tau_s(float(si), float(sdi), float(sddi))
            for si, sdi, sddi in zip(s, sdot, sddot)
        ])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        for j in range(tau.shape[1]):
            ax.plot(s, tau[:, j], linewidth=1.8, label=f'$\\tau_{j+1}$')
            if j < len(self.T_max):
                ax.axhline(self.T_max[j], color='k', linestyle='--', linewidth=0.8, alpha=0.45)
                ax.axhline(self.T_min[j], color='k', linestyle='--', linewidth=0.8, alpha=0.45)

        # Mark switching points on torque curves.
        marker_map = {
            'switch': ('o', 'Switch'),
            'singular': ('s', 'Singular'),
            'critical': ('^', 'Critical'),
            'tangency': ('D', 'Tangency'),
            'discontinuity': ('P', 'Discontinuity'),
            'regular': ('x', 'Regular-hit'),
        }
        shown = set()
        for i, s_i in enumerate(sw_s):
            sw_t = sw_types[i] if i < len(sw_types) else 'switch'
            marker, label = marker_map.get(sw_t, ('x', sw_t))
            lbl = label if label not in shown else None
            shown.add(label)
            for j in range(tau.shape[1]):
                tau_ij = float(np.interp(s_i, s, tau[:, j]))
                ax.scatter(
                    [s_i],
                    [tau_ij],
                    marker=marker,
                    s=38,
                    zorder=6,
                    label=lbl if j == 0 else None,
                )

        ax.set_xlabel('s')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Optimal Actuator Torques vs s')
        ax.grid(True, alpha=0.35)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig, ax, tau

    def plot_sddot(self, solution, ax=None):
        """
        Plot sddot(s) with switching-point markers.

        Parameters
        ----------
        solution : dict
            Output dictionary from solve().
        ax : matplotlib axis, optional
            Existing axis to draw on.

        Returns
        -------
        (fig, ax)
        """
        import matplotlib.pyplot as plt

        s = np.asarray(solution['s_traj'])
        sdot = np.asarray(solution['sdot_traj'])
        sddot = np.asarray(solution.get('sddot_traj'))
        sddot_min = np.asarray(solution.get('sddot_min_traj'))
        sddot_max = np.asarray(solution.get('sddot_max_traj'))
        sw_s = np.asarray(solution.get('switching_s', np.array([])))
        sw_types = list(solution.get('switching_types', []))

        if sddot is None or sddot.shape != sdot.shape:
            sddot = np.zeros_like(sdot)
        if sddot_min is None or sddot_min.shape != sdot.shape:
            sddot_min = np.full_like(sdot, np.nan)
        if sddot_max is None or sddot_max.shape != sdot.shape:
            sddot_max = np.full_like(sdot, np.nan)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.plot(s, sddot, linewidth=2.0, label='Optimal $\\ddot{s}(s)$')
        ax.plot(s, sddot_min, linestyle='--', linewidth=1.5, label='Lower bound $\\ddot{s}_{min}$')
        ax.plot(s, sddot_max, linestyle='--', linewidth=1.5, label='Upper bound $\\ddot{s}_{max}$')

        marker_map = {
            'switch': ('o', 'Switch'),
            'singular': ('s', 'Singular'),
            'critical': ('^', 'Critical'),
            'tangency': ('D', 'Tangency'),
            'discontinuity': ('P', 'Discontinuity'),
            'regular': ('x', 'Regular-hit'),
        }
        shown = set()
        for i, s_i in enumerate(sw_s):
            sw_t = sw_types[i] if i < len(sw_types) else 'switch'
            marker, label = marker_map.get(sw_t, ('x', sw_t))
            lbl = label if label not in shown else None
            shown.add(label)
            sddot_i = float(np.interp(s_i, s, sddot))
            ax.scatter([s_i], [sddot_i], marker=marker, s=55, label=lbl, zorder=5)

        ax.set_xlabel('s')
        ax.set_ylabel('$\\ddot{s}$')
        ax.set_title('PCTOM Acceleration and Switching Points')
        ax.grid(True, alpha=0.35)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig, ax

