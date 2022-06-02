from scipy import zeros_like
import rosbag
import pdb
import argparse
import pathlib
import mmint_utils
import scipy.stats

import numpy as np
from scipy.spatial.transform import Rotation as R

class FrankaArm:
    def __init__(self, cfg, bagfile, uses_netft=False):
        self.cfg = cfg
        self.arm_id = cfg["arm_id"]
        self.bagfile = bagfile
        self.uses_netft = uses_netft
        if self.uses_netft:
            self.netft_data = None
            self.netft_time = None
            self.F_T_EE = None
            self.F_T_EE_time = None
        else:
            self.O_F_ext = None
            self.O_F_ext_time = None
        
        self.W_T_O = None
        self.O_T_EE = None
        self.W_T_EE = None
        self.W_F_ext = None
        self.gripper_width = None

        self.O_T_EE_time = None
        self.W_T_EE_time = None
        self.W_F_ext_time = None
        self.gripper_width_time = None
        self.time_min = np.inf

        self.load_data()
    
    def load_data(self):
        self.load_W_T_O()

        if self.uses_netft:
            self.load_F_T_EE()
            self.load_O_T_EE()
            self.correct_O_T_EE()
            self.load_netft()
        else:
            self.load_O_T_EE()
            self.load_O_F_ext()
            self.load_EE_F_ext()

        self.load_W_F_ext()
        self.calibrate_F_ext()
        self.load_gripper_width()
        self.upsample_gripper_width()
        self.load_W_T_EE()

    def load_W_T_EE(self):
        self.W_T_EE = np.matmul(self.W_T_O, self.O_T_EE, axes=[(0, 1), (0, 1), (0, 1)])

    def load_O_T_TT(self):
        pass

    def load_O_T_EE(self):
        data_topic = self.cfg["state_topic"]
        self.O_T_EE = np.zeros((4, 4, self.bagfile.get_message_count(data_topic)))
        self.O_T_EE_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            raw_data = np.asarray(msg.O_T_EE)
            self.O_T_EE[:, :, i] = np.reshape(raw_data, (4, 4)).T
            self.O_T_EE_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.O_T_EE_time) < self.time_min:
            self.time_min = np.min(self.O_T_EE_time)

    def load_F_T_EE(self):
        data_topic = self.cfg["state_topic"]
        self.F_T_EE = np.zeros((4, 4, self.bagfile.get_message_count(data_topic)))
        self.F_T_EE_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            raw_data = np.asarray(msg.F_T_EE)
            self.F_T_EE[:, :, i] = np.reshape(raw_data, (4, 4)).T
            self.F_T_EE_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.F_T_EE_time) < self.time_min:
            self.time_min = np.min(self.F_T_EE_time)
    
    def correct_O_T_EE(self):
        z_offset = 0.1557244
        normal_z = .1034
        EE_T_F = np.linalg.inv(self.F_T_EE.T).T
        O_T_F = np.matmul(self.O_T_EE, EE_T_F, axes=[(0, 1), (0, 1), (0, 1)])
        self.F_T_TT = np.copy(self.F_T_EE)
        self.F_T_TT[2, -1, :] = z_offset
        self.F_T_EE[2, -1, :] = normal_z
        self.O_T_TT = np.matmul(O_T_F, self.F_T_TT, axes=[(0, 1), (0, 1), (0, 1)])
        self.O_T_EE = np.matmul(O_T_F, self.F_T_EE, axes=[(0, 1), (0, 1), (0, 1)])

    def load_W_T_O(self):
        self.W_T_O = np.zeros((4, 4))
        self.W_T_O[-1, -1] = 1
        data_topic = self.cfg["tf_topic"]
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            if len(msg.transforms) > 0:
                for tf in msg.transforms:
                    if tf.child_frame_id == f'{self.arm_id}_link0':
                        W_R_O = R.from_quat([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
                        self.W_T_O[:-1, :-1] = W_R_O.as_matrix()
                        self.W_T_O[:-1, -1] = np.asarray([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])

    def load_W_F_ext(self):
        EE_Adj_O = self.get_adjoint(np.linalg.inv(self.O_T_EE.T).T)            
        if self.uses_netft:
            self.downsample_netft()
            self.W_F_ext_time = self.aligned_netft_time
            W_O = np.squeeze(EE_Adj_O.transpose([2, 0, 1]) @ np.expand_dims(self.aligned_netft_data.T, -1)).T
        else:
            W_O = self.O_F_ext
            self.W_F_ext_time = self.O_F_ext_time
        O_Adj_W = self.get_adjoint(np.linalg.inv(self.W_T_O))
        self.W_F_ext = np.dot(O_Adj_W, W_O)

    def load_EE_F_ext(self):
        data_topic = self.cfg["state_topic"]
        self.EE_F_ext = np.zeros((6, self.bagfile.get_message_count(data_topic)))
        self.EE_F_ext_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            self.EE_F_ext[:, i] = -np.asarray(msg.K_F_ext_hat_K)
            self.EE_F_ext_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.EE_F_ext_time) < self.time_min:
            self.time_min = np.min(self.E_F_ext_time)

    def load_gripper_width(self):
        if self.uses_netft:
            self.gripper_width = self.cfg["gripper_width"]
        else:
            data_topic = self.cfg["gripper_topic"]
            self.gripper_width = np.zeros((self.bagfile.get_message_count(data_topic),))
            self.gripper_width_time = np.zeros((self.bagfile.get_message_count(data_topic),))
            i = 0
            for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
                self.gripper_width[i] += abs(msg.position[0])
                self.gripper_width[i] += abs(msg.position[1])
                self.gripper_width[i] /= 2
                self.gripper_width_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
                i += 1
            if np.min(self.gripper_width_time) < self.time_min:
                self.time_min = np.min(self.gripper_width_time)

    def upsample_gripper_width(self):
        self.aligned_gripper_width = np.zeros((self.O_T_EE.shape[-1],))
        if self.gripper_width_time is not None:
            for i in range(self.O_T_EE.shape[-1]):
                nearest_idx = (np.abs(self.O_T_EE_time[i] - self.gripper_width_time)).argmin()
                self.aligned_gripper_width[i] = self.gripper_width[nearest_idx]
        else:
            self.aligned_gripper_width[:] = self.gripper_width

    def load_netft(self):
        data_topic = self.cfg["netft_topic"]
        self.netft_data = np.zeros((6, self.bagfile.get_message_count(data_topic)))
        self.netft_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            self.netft_data[:, i] = np.asarray([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
            self.netft_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.netft_time) < self.time_min:
            self.time_min = np.min(self.netft_time)

    def downsample_netft(self):
        self.aligned_netft_data = np.zeros((6, self.O_T_EE.shape[-1]))
        self.aligned_netft_time = np.zeros_like(self.O_T_EE_time)
        for i in range(self.O_T_EE.shape[-1]):
            nearest_idx = (np.abs(self.O_T_EE_time[i] - self.netft_time)).argmin()
            self.aligned_netft_data[:, i] = self.netft_data[:, nearest_idx]
            self.aligned_netft_time[i] = self.netft_time[nearest_idx]

    def load_O_F_ext(self):
        data_topic = self.cfg["state_topic"]
        self.O_F_ext = np.zeros((6, self.bagfile.get_message_count(data_topic)))
        self.O_F_ext_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            self.O_F_ext[:, i] = -np.asarray(msg.O_F_ext_hat_K)
            self.O_F_ext_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.O_F_ext_time) < self.time_min:
            self.time_min = np.min(self.O_F_ext_time)

    def calibrate_F_ext(self):
        # calibration_data = self.W_F_ext[:, self.cfg["calibration"]["start"]:self.cfg["calibration"]["stop"]]
        calibration_data = self.EE_F_ext[:, self.cfg["calibration"]["start"]:self.cfg["calibration"]["stop"]]
        mean = np.zeros((calibration_data.shape[0],))
        var = np.zeros((calibration_data.shape[0],))
        for i in range(calibration_data.shape[0]):
            mean[i], var[i] = scipy.stats.distributions.norm.fit(calibration_data[i, :])
        if not self.uses_netft:
            # self.W_F_ext = (self.W_F_ext.T - mean).T
            self.EE_F_ext = (self.EE_F_ext.T - mean).T
        S_m = np.zeros((6, 6))
        np.fill_diagonal(S_m, var)
        self.S_m_inv = np.linalg.inv(S_m)

        
    def get_adjoint(self, a_H_b):
        if len(a_H_b.shape) == 2:
            Adj = np.zeros((6, 6))
            p = a_H_b[:-1, -1]
            skew_p = np.array([[    0, -p[2],  p[1]],
                            [ p[2],     0, -p[0]],
                            [-p[1],  p[0],    0]])
            Adj[:3, :3] = a_H_b[:-1, :-1].T
            Adj[3:, 3:] = a_H_b[:-1, :-1].T
            Adj[3:, :3] = -a_H_b[:-1, :-1].T @ skew_p
        else:
            Adj = np.zeros((6, 6, a_H_b.shape[-1]))
            p = a_H_b[:-1, -1]
            skew_p = np.array([[np.zeros(p[0].shape),                -p[2],                  p[1]],
                               [                p[2], np.zeros(p[0].shape),                 -p[0]],
                               [               -p[1],                 p[0], np.zeros(p[0].shape)]])
            a_R_b_T = np.transpose(a_H_b[:-1, :-1], (1, 0, 2))
            Adj[:3, :3] = a_R_b_T
            Adj[3:, 3:] = a_R_b_T
            Adj[3:, :3] = np.matmul(-a_R_b_T, skew_p, axes=[(0, 1), (0, 1), (0, 1)])
        return Adj
