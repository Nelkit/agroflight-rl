"""
Fixed (non-learned) inner controller: high-level velocity commands -> rotor thrusts.

The policy never touches the motors. It outputs four commands in the drone's
heading frame (forward velocity, lateral velocity, climb rate, yaw rate) and
this controller turns them into the four rotor thrusts the Skydio X2 expects.

    velocity error --P--> desired acceleration --> desired thrust vector
    thrust vector --> desired attitude --PD--> body torques
    [total thrust, torques] --mixer--> 4 rotor thrusts

There is deliberately no integral term and no position loop: a steady wind
therefore leaves a small velocity error, and correcting the resulting drift
back to the row centreline is the job the learned policy has to solve.
"""
import mujoco
import numpy as np


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class InnerController:
    def __init__(self, model, body_id, kv=3.0, kvz=5.0, att_wn=14.0, att_zeta=0.9,
                 yaw_rate_gain=8.0, yaw_hold_gain=2.0, max_tilt_deg=35.0):
        self.mass = float(model.body_subtreemass[body_id])
        self.gravity = float(-model.opt.gravity[2])
        self.thrust_low = model.actuator_ctrlrange[:, 0].copy()
        self.thrust_high = model.actuator_ctrlrange[:, 1].copy()

        # Inertia is stored in principal axes; rotate it into the body frame.
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[body_id])
        rot = rot.reshape(3, 3)
        inertia = rot @ np.diag(model.body_inertia[body_id]) @ rot.T
        self.inertia_diag = np.diag(inertia).copy()

        # Rotor arms relative to the centre of mass, and yaw-moment coefficients.
        arms, yaw_coef = [], []
        for i in range(1, 5):
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"thrust{i}")
            arms.append(model.site_pos[site] - model.body_ipos[body_id])
            yaw_coef.append(model.actuator_gear[i - 1, 5])
        arms = np.array(arms)
        # rows: total thrust, roll torque (y*F), pitch torque (-x*F), yaw torque
        mixer = np.array([np.ones(4), arms[:, 1], -arms[:, 0], np.array(yaw_coef)])
        self.mixer_inv = np.linalg.inv(mixer)

        self.kv, self.kvz = kv, kvz
        self.att_wn, self.att_zeta = att_wn, att_zeta
        self.yaw_rate_gain, self.yaw_hold_gain = yaw_rate_gain, yaw_hold_gain
        self.max_horizontal_accel = self.gravity * np.tan(np.deg2rad(max_tilt_deg))

    def thrusts(self, qpos, qvel, cmd, yaw_free):
        """cmd = [forward vel, lateral vel, climb rate (m/s), yaw rate (rad/s)]."""
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, qpos[3:7])
        rot = rot.reshape(3, 3)
        vel = qvel[0:3]
        omega = qvel[3:6]

        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        vel_des = np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], cmd[2]])

        accel = self.kv * (vel_des - vel)
        accel[2] = self.kvz * (vel_des[2] - vel[2])
        horizontal = np.linalg.norm(accel[:2])
        if horizontal > self.max_horizontal_accel:
            accel[:2] *= self.max_horizontal_accel / horizontal

        force = self.mass * (accel + np.array([0.0, 0.0, self.gravity]))
        total_thrust = max(float(force @ rot[:, 2]), 0.0)

        b3_des = force / max(np.linalg.norm(force), 1e-6)
        b2_des = np.cross(b3_des, np.array([c, s, 0.0]))
        b2_des /= max(np.linalg.norm(b2_des), 1e-6)
        rot_des = np.column_stack([np.cross(b2_des, b3_des), b2_des, b3_des])

        err = 0.5 * (rot_des.T @ rot - rot.T @ rot_des)
        att_err = np.array([err[2, 1], err[0, 2], err[1, 0]])
        torque = -self.inertia_diag * (self.att_wn ** 2 * att_err
                                       + 2 * self.att_zeta * self.att_wn * omega)

        yaw_rate_cmd = cmd[3] if yaw_free else -self.yaw_hold_gain * _wrap(yaw)
        torque[2] = self.inertia_diag[2] * self.yaw_rate_gain * (yaw_rate_cmd - omega[2])

        rotor = self.mixer_inv @ np.array([total_thrust, torque[0], torque[1], torque[2]])
        return np.clip(rotor, self.thrust_low, self.thrust_high)
