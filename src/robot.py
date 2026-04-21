"""
Defines the robot and related methods
"""

# Imports
import numpy as np


class Robot:
    def __init__(self, robot_type):
        '''
        Initialize the robot model based on configuration
        '''
        self.robot_type = robot_type
        if self.robot_type not in {'ELBOW_3DOF', 'CAR_1DOF'}:
            raise NotImplementedError(f"Robot type '{self.robot_type}' is not implemented in this code snippet.")

    def mass_matrix(self, q):
        if self.robot_type == 'CAR_1DOF':
            return np.array([[1.0]], dtype=float)

        q1, q2, q3 = q

        c2 = np.cos(2*q2)
        c3 = np.cos(q3)
        c23 = np.cos(2*q2 + q3)
        c23_2 = np.cos(2*q2 + 2*q3)

        M11 = 1.89*c3 + 8.4455 + 2.4295*c2 + 1.89*c23 + 0.896*c23_2
        M22 = 3.78*c3 + 6.7771
        M23 = 1.89*c3 + 1.812
        M33 = 1.812

        M = np.array([
            [M11, 0.0, 0.0],
            [0.0, M22, M23],
            [0.0, M23, M33]
        ])

        return M


    def coriolis(self, q, qdot):
        if self.robot_type == 'CAR_1DOF':
            return np.array([[0.0]], dtype=float)

        q1, q2, q3 = q
        q1d, q2d, q3d = qdot

        s23 = np.sin(2*q2 + q3)
        s23_2 = np.sin(2*q2 + 2*q3)
        s2 = np.sin(2*q2)
        s3 = np.sin(q3)

        # Precompute common terms
        a = 1.89*s23 + 0.896*s23_2 + 2.4295*s2
        b = 0.945*s23 + 0.896*s23_2 + 0.945*s3

        C = np.zeros((3, 3))

        # Row 1
        C[0, 0] = -a*q2d - b*q3d
        C[0, 1] = -a*q1d
        C[0, 2] = -b*q1d

        # Row 2
        C[1, 0] = a*q1d
        C[1, 1] = 0.0
        C[1, 2] = -1.89*s3*q3d

        # Row 3
        C[2, 0] = b*q1d
        C[2, 1] = -1.89*s3*q2d
        C[2, 2] = 0.0

        return C


    def gravity(self, q):
        if self.robot_type == 'CAR_1DOF':
            return np.array([0.0], dtype=float)

        q1, q2, q3 = q

        c2 = np.cos(q2)
        c23 = np.cos(q2 + q3)

        G = np.array([
            0.0,
            66.5118*c2 + 24.7212*c23,
            24.7212*c23
        ])

        return G

    def n_joints(self):
        '''
        Return the number of joints
        '''
        if self.robot_type == 'CAR_1DOF':
            return 1
        return 3

    def forward_kinematics(self, q):
        '''
        Compute the forward kinematics for joint configuration q
        '''
        if self.robot_type == 'CAR_1DOF':
            q = np.asarray(q, dtype=float).reshape(-1)
            return np.array([q[0]], dtype=float)

        # Forward kinematics for 3-DoF elbow manipulator
        theta1, theta2, theta3 = q

        d1 = 0.3585
        d2 = -0.037
        a1 = 0.050
        a2 = 0.300
        a3 = 0.250

        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)

        r = a1 + a2*c2 + a3*c23

        x = c1 * r
        y = s1 * r
        z = d1 + d2 + a2*s2 + a3*s23

        return np.array([x, y, z])

    def inverse_kinematics(self, r, q0=None, elbow_up=True):
        '''
        Compute the inverse kinematics for position r.

        Parameters
        ----------
        r : array-like, shape (3,)
            Target position.
        q0 : array-like, optional
            Seed joint configuration. If provided, both elbow branches are
            evaluated and the solution closest to q0 is returned.
        elbow_up : bool
            Preferred branch when q0 is not provided.
        '''
        if self.robot_type == 'CAR_1DOF':
            r = np.asarray(r, dtype=float).reshape(-1)
            return np.array([r[0]], dtype=float)

        x, y, z = r
        d1 = 0.3585
        d2 = -0.037
        a1 = 0.050
        a2 = 0.300
        a3 = 0.250

        # theta1
        theta1 = np.arctan2(y, x)

        # planar reduction
        r = np.sqrt(x**2 + y**2) - a1
        z_ = z - (d1 + d2)

        # theta3
        D = (r**2 + z_**2 - a2**2 - a3**2) / (2 * a2 * a3)

        # numerical safety
        D = np.clip(D, -1.0, 1.0)

        def _solve_with_sign(sign):
            th3 = np.arctan2(sign * np.sqrt(1 - D**2), D)
            th2 = np.arctan2(z_, r) - np.arctan2(a3*np.sin(th3), a2 + a3*np.cos(th3))
            return np.array([theta1, th2, th3], dtype=float)

        q_up = _solve_with_sign(+1.0)
        q_dn = _solve_with_sign(-1.0)

        if q0 is not None:
            q0 = np.asarray(q0, dtype=float).reshape(-1)
            # choose branch closest to seed in wrapped angular distance
            def _wrapped_norm(q):
                d = np.arctan2(np.sin(q - q0), np.cos(q - q0))
                return np.linalg.norm(d)
            return q_up if _wrapped_norm(q_up) <= _wrapped_norm(q_dn) else q_dn

        return q_up if elbow_up else q_dn

    def jacobian(self, q):
        '''
        Compute the Jacobian at joint configuration q
        '''
        if self.robot_type == 'CAR_1DOF':
            return np.array([[1.0]], dtype=float)

        theta1, theta2, theta3 = q
        d1 = 0.3585
        d2 = -0.037
        a1 = 0.050
        a2 = 0.300
        a3 = 0.250
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)
        J = np.zeros((3, 3))
        J[0, 0] = -s1 * (a1 + a2*c2 + a3*c23)
        J[0, 1] = -c1 * (a2*s2 + a3*s23)
        J[0, 2] = -c1 * a3 * s23
        J[1, 0] = c1 * (a1 + a2*c2 + a3*c23)
        J[1, 1] = -s1 * (a2*s2 + a3*s23)
        J[1, 2] = -s1 * a3 * s23
        J[2, 0] = 0.0
        J[2, 1] = a2*c2 + a3*c23
        J[2, 2] = a3 * c23
        return J

    def jacobian_derivative(self, q, dq, _eps=1e-6):
        '''
        Compute the directional derivative of the Jacobian along joint direction dq,
        numerically via finite difference:
            dJ ≈ (J(q + dq·_eps) - J(q)) / _eps ; _eps is a small step size
        Returns the position-only (first 3 rows) result.
        '''
        if self.robot_type == 'CAR_1DOF':
            return np.array([[0.0]], dtype=float)

        J0 = self.jacobian(q)
        J1 = self.jacobian(q + dq * _eps)
        return (J1 - J0) / _eps

    def torque(self, q, dq, ddq):
        '''
        Compute the torque τ = M(q)ddq + C(q, dq)dq + G(q)
        '''
        if self.robot_type == 'CAR_1DOF':
            ddq = np.asarray(ddq, dtype=float).reshape(-1)
            return np.array([ddq[0]], dtype=float)

        M = self.mass_matrix(q)
        C = self.coriolis(q, dq)
        G = self.gravity(q)
        tau = M @ ddq + C @ dq + G
        return tau
    

