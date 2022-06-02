import argparse
import pathlib
import pickle
import random

from numpy import dtype
import rospy
import tf
import os
import pdb
from arc_utilities import reliable_tf
import mmint_utils
import cvxpy as cp 
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from experiment import CPFExperiment
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import inverse_matrix as inv

np.set_printoptions(suppress=True)

tf_listener = None
tf_broadcaster = None
tfw = None
pubs = None

def get_grid_points(res, shape):
    xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, shape), np.linspace(-0.5, 0.5, shape), np.linspace(-0.5, 0.5, shape))
    unscaled_points = np.vstack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))).T
    points = (unscaled_points*res).T
    return points

def get_surface_points(sdf, vg, upper, lower):
    points = get_grid_points(vg.scale, sdf.shape[0])
    d = sdf.flatten()
    d = d.reshape([1, -1])
    xyzd = np.vstack((points, d))
    surface_idx = np.argwhere(np.logical_and(xyzd[-1, :] > lower, xyzd[-1, :] < upper))[:, 0]
    surface_pts = np.vstack((xyzd[:-1, surface_idx], np.ones((1, surface_idx.shape[0]))))
    sdf_pts = np.vstack((xyzd[:-1, :], np.ones((1, xyzd.shape[-1])), xyzd[-1]))
    origin_offset = (np.array([vg.origin.x, vg.origin.y, vg.origin.z]) - np.min(surface_pts, axis=1)[:-1]).reshape(-1, 1)
    surface_pts[:-1] += origin_offset
    sdf_pts[:3] += origin_offset
    return surface_pts, surface_idx

def get_surface_normals(grad, surface_idx):
    surface_norms = grad.reshape(-1, grad.shape[-1]).T[:, surface_idx]
    return surface_norms

def get_gripper_points(cfg, gripper_width, W_T_EE):
    xmin = -cfg["fingertip"]/2
    xmax = cfg["fingertip"]/2
    zmin = -cfg["fingertip"]/2
    zmax = cfg["fingertip"]/2
    xx, zz = np.meshgrid(np.linspace(xmin, xmax, cfg["n_gripper_pts"]), np.linspace(zmin, zmax, cfg["n_gripper_pts"]))
    yy = gripper_width*np.ones((cfg["n_gripper_pts"]**2,))
    fingertip_l = np.vstack((xx.reshape(-1), yy, zz.reshape(-1), np.ones((cfg["n_gripper_pts"]**2,))))
    fingertip_r = np.vstack((xx.reshape(-1), -yy, zz.reshape(-1), np.ones((cfg["n_gripper_pts"]**2,))))
    fingertips = W_T_EE @ np.hstack((fingertip_l, fingertip_r))
    return fingertips

def get_adjoint(a_H_b):
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

def init_clp_pts(n_surf_pts, n_clp):
    clp_init_idx = random.sample(np.arange(n_surf_pts).tolist(), n_clp)
    return np.asarray(clp_init_idx)

def apply_clp_motion_model(surf_pts, clp_idx, max_step_size=0.01):
    dirs = np.random.uniform(-max_step_size, max_step_size, (3, clp_idx.size))
    new_pts = surf_pts[:-1, clp_idx] + dirs
    new_clp_idx = np.zeros_like(clp_idx)
    for i in range(clp_idx.size):
        new_clp_idx[i] = np.argmin(np.linalg.norm(surf_pts[:-1] - new_pts[:, i].reshape(-1, 1), axis=0))
    return new_clp_idx

def get_rot(a, b):
    # from a to b
    v = np.cross(a, b, axisb=0).T
    c = np.dot(a, b)
    rot = np.zeros((3, 3, c.shape[0]))
    skew_idx = np.where(c != -1)[0]
    eye_idx = np.where(c == -1)[0]
    skew_v = np.array([[     np.zeros((skew_idx.shape[0],)), -v[2, skew_idx],  v[1, skew_idx]],
                        [ v[2, skew_idx],     np.zeros((skew_idx.shape[0],)), -v[0, skew_idx]],
                        [-v[1, skew_idx],  v[0, skew_idx],    np.zeros((skew_idx.shape[0],))]])
    rot[:, :, skew_idx] = np.eye(3).reshape(3, 3, 1) + skew_v + np.matmul(skew_v, skew_v, axes=[(0, 1), (0, 1), (0, 1)])*(1/(1 + c[skew_idx]))
    if eye_idx.shape[0] != 0:
        rot[:, :, eye_idx] = -np.eye(3).reshape(3, 3, 1)
    return rot

def get_perpendicular(n):
    if np.sum(n) == 0.0:
        return None
    if n[0] == 0.0:
        return np.array([1, 0, 0])
    if n[1] == 0.0:
        return np.array([0, 1, 0])
    if n[2] == 0.0:
        return np.array([0, 0, 1])
    return np.array([1, 1, -(n[0] + n[1])/n[2]])

def friction_cone_approx(n_approx, mu=0.8):
    f = -np.ones((3, n_approx))
    for i in range(n_approx):
        f[0, i] = mu*np.sin(2*np.pi*i/n_approx)
        f[1, i] = mu*np.cos(2*np.pi*i/n_approx)
    return f

def low_variance_resample(particles, weights):
    resampled_particles = np.zeros((particles.size,), dtype=int)
    r = random.uniform(0, 1/particles.size)
    c = weights[0]
    i = 0
    resample_count = 0
    for m in range(0, particles.shape[0]):
        U = r + (m - 1)*(1/particles.shape[0])
        while U > c:
            i += 1
            c += weights[i]
        resampled_particles[resample_count] = particles[i]
        resample_count += 1
    return resampled_particles

def get_vec_rot(z_axis, surf_norms):
    unit_surf_norms = surf_norms/np.linalg.norm(surf_norms, axis=0)
    R_c = get_rot(z_axis, unit_surf_norms)
    return R_c

def apply_cpf_measurement_model(cfg, ns, contact_pts, contact_norms, F_obj, S_m_inv, W_T_obj, viz=False):
    W_R_c = get_vec_rot(np.asarray([0, 0, 1]), contact_norms)
    W_T_c = np.zeros((4, 4, cfg["n_clp"]))
    W_T_c[:-1, :-1, :] = W_R_c
    W_T_c[:, -1, :] = contact_pts
    c_fc = friction_cone_approx(cfg["n_fc"])
    c_Adj_obj = np.zeros((6, 6, cfg["n_clp"]))
    obj_T_c = np.matmul(np.linalg.inv(W_T_obj), W_T_c, axes=[(0, 1), (0, 1), (0, 1)])
    c_Adj_obj = get_adjoint(np.linalg.inv(obj_T_c.T).T)
    Beta = np.matmul(c_Adj_obj, np.vstack((c_fc, np.zeros_like(c_fc))), axes=[(0, 1), (0, 1), (0, 1)])
    Q = np.matmul(Beta.transpose([1, 0, 2]), np.matmul(S_m_inv, Beta, axes=[(0, 1), (0, 1), (0, 1)]), axes=[(0, 1), (0, 1), (0, 1)])
    P = np.matmul(F_obj.reshape(1, -1), np.matmul(S_m_inv, Beta, axes=[(0, 1), (0, 1), (0, 1)]), axes=[(0, 1), (0, 1), (0, 1)]).reshape(-1, cfg["n_clp"])
    A = np.eye(cfg["n_fc"])
    b = np.zeros((cfg["n_fc"],))
    alpha = cp.Variable(cfg["n_fc"])
    raw_losses = np.zeros((cfg["n_clp"],))
    est_alpha = np.zeros((8, cfg["n_clp"]))
    for i in range(cfg["n_clp"]):
        try:
            prob = cp.Problem(cp.Minimize((1/2)*cp.QuadForm(alpha, Q[:, :, i]) + P[:, i] @ alpha + (1/2)*F_obj @ S_m_inv @ F_obj), [A @ alpha <= b])
        except:
            pdb.set_trace()
        prob.solve()
        raw_losses[i] = prob.value
        est_alpha[:, i] = alpha.value
    
    losses = np.exp(cfg["spread_param"]*raw_losses/cfg["QP_scaler"])
    normed_losses = losses/np.sum(losses)

    # est_wrench_c = np.vstack((np.dot(est_alpha.T, c_fc.T).T, np.zeros((3, cfg["n_clp"]))))
    est_wrench_c = np.vstack((np.dot(-est_alpha.T, c_fc.T).T, np.zeros((3, cfg["n_clp"]))))
    est_wrench_obj = np.squeeze(np.matmul(c_Adj_obj, np.expand_dims(est_wrench_c, 1), axes=[(0, 1), (0, 1), (0, 1)]))

    # if viz:
    #     max_idx = np.argmax(normed_losses)
    #     send_tf(W_T_c[:, :, max_idx], "tool_clp", "world")
    #     est_F_obj = create_force_arrow(est_wrench_obj[:3, max_idx], "tool", rgba=(0.66, 0.8, 0.2, 1.0))
    #     pubs["tool_est_fobj"].publish(est_F_obj)
    #     gt_F_obj = create_force_arrow(F_obj[:3], "tool", rgba=(0.66, 0.2, 0.92, 1.0))
    #     pubs["tool_fobj_gt"].publish(gt_F_obj)
    #     est_F_c = create_force_arrow(est_wrench_c[:3, max_idx], "tool_clp", rgba=(0.66, 0.8, 0.2, 1.0))
    #     pubs["tool_est_fc"].publish(est_F_c)
    #     fc_cloud = create_fc_cloud("tool_clp", c_fc)
    #     pubs["tool_norms"].publish(fc_cloud)

    pc = create_point_cloud_intensity(contact_pts, intensity=normed_losses)
    pubs[f'{ns}_clps'].publish(pc)
    # pdb.set_trace()
    return raw_losses, normed_losses, est_wrench_obj

def step_cpf(cfg, ns, F_obj, S_m_inv, W_T_obj, last_clp_idx, surf_pts, surf_norms, viz=False):
    clp_idx = apply_clp_motion_model(surf_pts, last_clp_idx, max_step_size=cfg["max_step_size"])
    raw_losses, normed_losses, est_wrench_obj = apply_cpf_measurement_model(cfg, ns, surf_pts[:, clp_idx], surf_norms[:, clp_idx], F_obj, S_m_inv, W_T_obj, viz)
    updated_clp_idx = low_variance_resample(clp_idx, normed_losses)
    return updated_clp_idx, clp_idx, normed_losses, est_wrench_obj

def load_object(cfg, arm_id):
    with open(cfg[arm_id]["object"]["sdf_path"], "rb") as in_f:
        sdf = pickle.load(in_f)

    with open(cfg[arm_id]["object"]["vg_path"], "rb") as in_f:
        vg = pickle.load(in_f)

    with open(cfg[arm_id]["object"]["grad_path"], "rb") as in_f:
        grad = pickle.load(in_f)    
    
    surf_pts, surf_idx = get_surface_points(sdf, vg, cfg[arm_id]["object"]["sdf_upper"], cfg[arm_id]["object"]["sdf_lower"])
    surf_norms = get_surface_normals(grad, surf_idx)
    
    return surf_pts, surf_norms

def create_point_cloud_intensity(pts, intensity, frame="world"):
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]
    xyzi_matrix = np.concatenate((pts[:3], intensity.reshape(1,-1)), 0).T
    xyzi_list = [tuple(xyzi) for xyzi in xyzi_matrix]
    header = Header()
    header.frame_id = frame
    header.stamp = rospy.Time.now()
    pc2 = point_cloud2.create_cloud(header, fields, xyzi_list)
    return pc2

def create_point_cloud(pts):
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
    header = Header()
    header.frame_id = "world"
    header.stamp = rospy.Time.now()
    try:
        pc2 = point_cloud2.create_cloud(header, fields, pts[:-1].T)
    except:
        pc2 = point_cloud2.create_cloud(header, fields, pts[:-1].reshape((1, -1)))
    return pc2

def create_fc_cloud(frame_id, fc_vectors, mag=0.02, alpha=0.5):
    fc_cloud = MarkerArray()
    unit_norms = fc_vectors/np.linalg.norm(fc_vectors, axis=0)
    for i in range(fc_vectors.shape[-1]):
        fc_arrow = Marker()
        fc_arrow.header = Header()
        fc_arrow.header.stamp = rospy.Time.now()
        fc_arrow.header.frame_id = frame_id
        fc_arrow.ns = "norm"
        fc_arrow.id = i
        fc_arrow.type = 0
        fc_arrow.action = 0
        fc_arrow.pose.orientation.w = 1.0
        start_point = Point()
        (start_point.x, start_point.y, start_point.z) = (0.0, 0.0, 0.0)
        end_point = Point()
        end_point.x = start_point.x - mag*unit_norms[0, i]
        end_point.y = start_point.y - mag*unit_norms[1, i]
        end_point.z = start_point.z - mag*unit_norms[2, i]
        fc_arrow.points = [start_point, end_point]
        (fc_arrow.scale.x, fc_arrow.scale.y, fc_arrow.scale.z) = np.asarray([0.001, 0.003, 0.003])
        (fc_arrow.color.r, fc_arrow.color.g, fc_arrow.color.b, fc_arrow.color.a) = np.asarray([0, 0, 0, alpha])
        fc_arrow.lifetime = rospy.Duration(0)
        fc_cloud.markers.append(fc_arrow)
    return fc_cloud

def create_norm_cloud(W_T_obj, pts, norms, n=100, mag=0.005, alpha=1.0):
    norm_cloud = MarkerArray()
    clear_cloud = MarkerArray()
    sample_idx = np.random.choice(np.arange(pts.shape[-1]), n, False)
    W_R_obj = W_T_obj[:-1, :-1]
    tf_pts = W_T_obj @ pts
    tf_norms = W_R_obj @ norms
    unit_tf_norms = tf_norms/np.linalg.norm(tf_norms, axis=0)
    for i in range(n):
        norm_arrow = Marker()
        norm_arrow.header = Header()
        norm_arrow.header.stamp = rospy.Time.now()
        clear_arrow = Marker()
        clear_arrow.header = Header()
        clear_arrow.header.stamp = rospy.Time.now()
        norm_arrow.header.frame_id = "world"
        norm_arrow.ns = "norm"
        norm_arrow.id = sample_idx[i]
        norm_arrow.type = 0
        norm_arrow.action = 0
        clear_arrow.action = 3
        norm_arrow.pose.orientation.w = 1.0
        start_point = Point()
        (start_point.x, start_point.y, start_point.z) = tf_pts[:-1, sample_idx[i]]
        end_point = Point()
        end_point.x = start_point.x + mag*unit_tf_norms[0, sample_idx[i]]
        end_point.y = start_point.y + mag*unit_tf_norms[1, sample_idx[i]]
        end_point.z = start_point.z + mag*unit_tf_norms[2, sample_idx[i]]
        norm_arrow.points = [start_point, end_point]
        (norm_arrow.scale.x, norm_arrow.scale.y, norm_arrow.scale.z) = np.asarray([0.0001, 0.0003, 0.0003])
        (norm_arrow.color.r, norm_arrow.color.g, norm_arrow.color.b, norm_arrow.color.a) = np.asarray([0, 0, 0, alpha])
        norm_arrow.lifetime = rospy.Duration(0)
        norm_arrow.frame_locked = True
        norm_cloud.markers.append(norm_arrow)
        clear_cloud.markers.append(clear_arrow)
    return clear_cloud, norm_cloud

def create_force_arrow(F_obj, frame_id, rgba=(0.0, 0.0, 0.0, 1.0), mag=0.02):
    force_arrow = Marker()
    unit_force_vector = F_obj[:3]/np.linalg.norm(F_obj[:3])
    force_arrow.header = Header()
    force_arrow.header.stamp = rospy.Time.now()
    force_arrow.header.frame_id = frame_id
    force_arrow.ns = "force"
    force_arrow.id = 0
    force_arrow.type = 0
    force_arrow.action = 0
    force_arrow.pose.orientation.w = 1.0
    start_point = Point()
    start_point.x = -mag*unit_force_vector[0]
    start_point.y = -mag*unit_force_vector[1]
    start_point.z = -mag*unit_force_vector[2]
    end_point = Point()
    (end_point.x, end_point.y, end_point.z) = np.zeros((3,))
    force_arrow.points = [start_point, end_point]
    (force_arrow.scale.x, force_arrow.scale.y, force_arrow.scale.z) = np.asarray([0.001, 0.003, 0.003])
    (force_arrow.color.r, force_arrow.color.g, force_arrow.color.b, force_arrow.color.a) = rgba
    force_arrow.lifetime = rospy.Duration(0)
    force_arrow.frame_locked = True
    return force_arrow

def send_tf(tf_mat, child, parent):
    tfw.start_send_transform(tf_mat[:-1, -1], tf.transformations.quaternion_from_matrix(tf_mat), parent, child)
    
def create_pub_dict():
    global pubs
    pub_dict = {}
    for pub in pubs:
        pub_dict[pub.name[1:]] = pub
    pubs = pub_dict

def start_ros(trial, signal_idx):
    global tf_listener, tf_broadcaster, pubs, tfw
    rospy.init_node("cpfgrasp")
    r = rospy.Rate(10)

    pubs = []
    pubs.append(rospy.Publisher('/joint_states', JointState, queue_size=10))
    pubs.append(rospy.Publisher('/tool_cloud', PointCloud2, queue_size=10))
    pubs.append(rospy.Publisher('/tool_norms', MarkerArray, queue_size=10))
    pubs.append(rospy.Publisher('/tool_clps', PointCloud2, queue_size=10))
    pubs.append(rospy.Publisher('/tool_est_fobj', Marker, queue_size=10))
    pubs.append(rospy.Publisher('/tool_est_fc', Marker, queue_size=10))
    pubs.append(rospy.Publisher('/tool_fobj_gt', Marker, queue_size=10))
    pubs.append(rospy.Publisher('/viz', Marker, queue_size=10))
    for pub in pubs:
        while pub.get_num_connections() < 1:
            r.sleep()
    pubs[0].publish(trial.joint_states[signal_idx])

    tfw = reliable_tf.ReliableTF()
    
    create_pub_dict()
    return r

def save_results(out_path, cfg, t_cl, t_losses, t_clp_idx):
    n_exp = 0
    exp_exists = os.path.isdir(f'{out_path}/t{n_exp}')
    while exp_exists:
        n_exp += 1
        exp_exists = os.path.isdir(f'{out_path}/t{n_exp}')
    os.mkdir(f'{out_path}/t{n_exp}')
    np.save(f'{out_path}/t{n_exp}/t_cl.npy', t_cl)
    np.save(f'{out_path}/t{n_exp}/t_losses.npy', t_losses)
    np.save(f'{out_path}/t{n_exp}/t_clp_idx.npy', t_clp_idx)
    cfg["out_path"] = f'{out_path}/t{n_exp}'
    mmint_utils.dump_cfg(f'{out_path}/t{n_exp}/cfg.yaml', cfg)

def viz_pt(point, frame):
    tstpt_marker = Marker()
    tstpt_marker.header = Header()
    tstpt_marker.header.frame_id = frame
    tstpt_marker.type = 2
    tstpt_marker.pose.position.x = point[0]
    tstpt_marker.pose.position.y = point[1]
    tstpt_marker.pose.position.z = point[2]
    tstpt_marker.pose.orientation.w = 1.0
    tstpt_marker.scale.x = 0.005
    tstpt_marker.scale.y = 0.005
    tstpt_marker.scale.z = 0.005
    tstpt_marker.color.r = 0.0
    tstpt_marker.color.g = 0.0
    tstpt_marker.color.b = 0.0
    tstpt_marker.color.a = 1.0
    pubs["viz"].publish(tstpt_marker)

def main(cfg, trial):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # signal_idx = random.randint(cfg["contact"]["start"], cfg["contact"]["stop"])
    signal_idx = cfg["contact"]["start"]
    cfg["signal_idx"] = signal_idx
    
    r = start_ros(trial, signal_idx)
    # load and prepare grasped object models
    tool_surf_pts, tool_surf_norms = load_object(cfg, "arm_2")
    W_T_t_gt = trial.arm_2.W_T_EE[:, :, signal_idx]

    send_tf(W_T_t_gt, "tool", "world")

    tf_tool_surf_pts = W_T_t_gt @ tool_surf_pts
    tf_tool_surf_norms = W_T_t_gt[:-1, :-1] @ tool_surf_norms

    tool_cloud = create_point_cloud(tf_tool_surf_pts)
    pubs["tool_cloud"].publish(tool_cloud)
    
    t_cl = np.zeros((4, cfg["n_clp"], cfg["cpf_steps"]))

    t_clp_losses = np.zeros((cfg["n_clp"], cfg["cpf_steps"]))

    tool_clp_idx = np.zeros((cfg["n_clp"], cfg["cpf_steps"] + 1), dtype=int)
    tool_clp_idx[:, 0] = init_clp_pts(tf_tool_surf_pts.shape[-1], n_clp=cfg["n_clp"])

    # W_Adj_t = trial.arm_2.get_adjoint(W_T_t_gt)
    # F_t = W_Adj_t @ trial.arm_2.W_F_ext[:, signal_idx]
    F_t = trial.arm_2.EE_F_ext

    gt = np.array([0.27957, 0.010878, 0.66636+0.025])
    
    # viz_pt(gt, "world")
    # pdb.set_trace()
    for cpf_step in range(cfg["cpf_steps"]):
        t_updated_clp, t_clp, t_losses, t_wrench = step_cpf(cfg, "tool", F_t, trial.arm_2.S_m_inv, W_T_t_gt, tool_clp_idx[:, cpf_step], tf_tool_surf_pts, tf_tool_surf_norms, viz=True)
        tool_clp_idx[:, cpf_step + 1] = t_updated_clp
        t_cl[:, :, cpf_step] = tf_tool_surf_pts[:, t_clp]
        t_clp_losses[:, cpf_step] = t_losses
        rospy.sleep(0.5)
        # pdb.set_trace()
    # pdb.set_trace()
    # save_results(cfg["out_path"], cfg, t_cl, t_clp_losses, tool_clp_idx)
    tfw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='the .yaml file with config params', type=pathlib.Path)
    parser.add_argument('seed', help="random seed", type=int)
    args = parser.parse_args()

    cfg = mmint_utils.load_cfg(args.cfg)
    cfg["out_path"] = "../out/cpfgrasp"
    cfg["seed"] = args.seed
    trial = CPFExperiment(cfg)
    main(cfg, trial)
