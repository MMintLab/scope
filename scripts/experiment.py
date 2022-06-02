import rosbag
import pdb
import argparse
import pathlib
import mmint_utils

import numpy as np
import matplotlib.pyplot as plt

from manual_cpf_data import FrankaArm
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bagfile = rosbag.Bag(self.cfg["bag_path"])
        self.arm_1 = FrankaArm(self.cfg["arm_1"], self.bagfile)
        self.arm_2 = FrankaArm(self.cfg["arm_2"], self.bagfile, uses_netft=True)
        self.load_joint_states()
        self.update_time_data()
        self.load_W_contact_pt()
        # self.viz_wrench_data()

    def update_time_data(self):
        global_time_min = min(self.arm_1.time_min, self.arm_2.time_min)
        arm_1_data = list(self.arm_1.__dict__.keys())
        for data in arm_1_data:
            if data != "time_min" and "time" in data:
                old_time = getattr(self.arm_1, data)
                if old_time is not None:
                    new_time = old_time - global_time_min
                    setattr(self.arm_1, data, new_time)
        arm_2_data = list(self.arm_2.__dict__.keys())
        for data in arm_2_data:
            if data != "time_min" and "time" in data:
                old_time = getattr(self.arm_2, data)
                if old_time is not None:
                    new_time = old_time - global_time_min
                    setattr(self.arm_2, data, new_time)
        self.joint_states_time = self.aligned_js_time -global_time_min
        self.joint_states = self.aligned_js_data
    
    def load_W_contact_pt(self):
        self.W_contact_pt = np.matmul(self.arm_2.W_T_O, self.arm_2.O_T_TT, axes=[(0, 1), (0, 1), (0, 1)])

    def load_joint_states(self):
        data_topic = "/combined_panda/joint_states"
        self.joint_states = []
        self.joint_states_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            msg.name.extend(['panda_2_finger_joint1', 'panda_2_finger_joint2'])
            list_position = list(msg.position)
            list_position.extend([msg.position[7], msg.position[8]])
            msg.position = tuple(list_position)
            self.joint_states.append(msg)
            self.joint_states_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.joint_states_time) < self.arm_1.time_min or np.min(self.joint_states_time) < self.arm_2.time_min:
            self.arm_1.time_min = np.min(self.joint_states_time)
            self.arm_2.time_min = np.min(self.joint_states_time)
        self.downsample_joint_states()

    def downsample_joint_states(self):
        self.aligned_js_data = []
        self.aligned_js_time = np.zeros_like(self.arm_1.O_T_EE_time)
        for i in range(self.arm_1.O_T_EE.shape[-1]):
            nearest_idx = (np.abs(self.arm_1.O_T_EE_time[i] - self.joint_states_time)).argmin()
            self.aligned_js_data.append(self.joint_states[nearest_idx])
            self.aligned_js_time[i] = self.joint_states_time[nearest_idx]

    def viz_wrench_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("Holding Arm")
        ax.plot(self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[0, :], 'r', self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[1, :], 'g', self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[2, :], 'b')
        ax = fig.add_subplot(223)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (Nm)")
        ax.plot(self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[3, :], 'r', self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[4, :], 'g', self.arm_1.W_F_ext_time, self.arm_1.W_F_ext[5, :], 'b')
        ax = fig.add_subplot(222)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("Poking Arm")
        ax.plot(np.arange(self.arm_2.W_F_ext_time.shape[0]), self.arm_2.W_F_ext[1, :], 'r')#, self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[1, :], 'g', self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[2, :], 'b')
        ax = fig.add_subplot(224)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (Nm)")
        ax.plot(self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[3, :], 'r', self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[4, :], 'g', self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[5, :], 'b')
        fig.suptitle("Proprioceptive Feedback", fontsize=32)
        plt.show()

class CPFExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bagfile = rosbag.Bag(self.cfg["bag_path"])
        self.contact_idx = self.cfg["contact"]["start"]
        self.arm_2 = FrankaArm(self.cfg["arm_2"], self.bagfile, self.contact_idx)
        self.load_joint_states()
        self.update_time_data()
        # self.viz_wrench_data()

    def update_time_data(self):
        global_time_min = self.arm_2.time_min
        arm_2_data = list(self.arm_2.__dict__.keys())
        for data in arm_2_data:
            if data != "time_min" and "time" in data:
                old_time = getattr(self.arm_2, data)
                if old_time is not None:
                    new_time = old_time - global_time_min
                    setattr(self.arm_2, data, new_time)
        self.joint_states_time = self.aligned_js_time -global_time_min
        self.joint_states = self.aligned_js_data
    
    def load_joint_states(self):
        data_topic = "/combined_panda/joint_states"
        self.joint_states = []
        self.joint_states_time = np.zeros((self.bagfile.get_message_count(data_topic),))
        i = 0
        for _, msg, _ in self.bagfile.read_messages(topics=data_topic):
            list_position = list(msg.position)
            list_position[-2:] = (0.005, 0.005)
            msg.position = tuple(list_position)
            self.joint_states.append(msg)
            self.joint_states_time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs/1E9
            i += 1
        if np.min(self.joint_states_time) < self.arm_2.time_min:
            self.arm_2.time_min = np.min(self.joint_states_time)
        self.downsample_joint_states()

    def downsample_joint_states(self):
        self.aligned_js_data = []
        self.aligned_js_time = np.zeros_like(self.arm_2.O_T_EE_time)
        for i in range(self.arm_2.O_T_EE.shape[-1]):
            nearest_idx = (np.abs(self.arm_2.O_T_EE_time[i] - self.joint_states_time)).argmin()
            self.aligned_js_data.append(self.joint_states[nearest_idx])
            self.aligned_js_time[i] = self.joint_states_time[nearest_idx]

    def viz_wrench_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("Hex Arm")
        ax.plot(np.arange(self.arm_2.W_F_ext_time.shape[0]), self.arm_2.W_F_ext[0, :], 'r', np.arange(self.arm_2.W_F_ext_time.shape[0]), self.arm_2.W_F_ext[1, :], 'g', np.arange(self.arm_2.W_F_ext_time.shape[0]), self.arm_2.W_F_ext[2, :], 'b')
        ax = fig.add_subplot(212)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (Nm)")
        ax.plot(self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[3, :], 'r', self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[4, :], 'g', self.arm_2.W_F_ext_time, self.arm_2.W_F_ext[5, :], 'b')
        fig.suptitle("Proprioceptive Feedback", fontsize=32)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='the .yaml file with config params', type=pathlib.Path)
    args = parser.parse_args()

    cfg = mmint_utils.load_cfg(args.cfg)
    exp_1 = CPFExperiment(cfg)
    pdb.set_trace()
