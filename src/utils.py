'''
utility classes and methods
'''

# Imports
import numpy as np

class PathParametrize:
    """
    Handles geometric path parametrization and its derivatives.
    """
    def __init__(self, x0, x1, method='linear', robot=None):
        self.x0 = np.array(x0) if not isinstance(x0, np.ndarray) else x0
        self.x1 = np.array(x1) if not isinstance(x1, np.ndarray) else x1
        self.method = method
        self.robot = robot

        # Precompute params based on method
        if self.method == 'linear':
            self.delta = self.x1 - self.x0
            self.r = lambda s: self.x0 + s * self.delta
            self.dr_ds = self.delta
            self.d2r_ds2 = np.zeros_like(self.dr_ds)
        elif self.method == 'joint_linear':
            # Linear in joint space
            # Inputs are q0 and q1
            self.delta = self.x1 - self.x0
            self.r = lambda s: self.robot.forward_kinematics(self.x0 + s * self.delta)
            self.dr_ds = lambda s: self.robot.jacobian(self.x0 + s * self.delta) @ self.delta
            self.d2r_ds2 = lambda s: self.robot.jacobian_derivative(self.x0 + s * self.delta, self.delta) @ self.delta
        elif self.method == 'cubic':
            raise NotImplementedError("Cubic interpolation not implemented yet")

    def compute_r_s(self, s):
        """
        Compute the position in task space for a given path parameter s
        """
        return self.r(s)
    def compute_dr_ds(self, s):
        """
        Compute the velocity in task space for a given path parameter s
        """
        return self.dr_ds(s) if callable(self.dr_ds) else self.dr_ds
    def compute_d2r_ds2(self, s):
        """
        Compute the acceleration in task space for a given path parameter s
        """
        return self.d2r_ds2(s) if callable(self.d2r_ds2) else self.d2r_ds2