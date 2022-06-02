import binvox_rw
import argparse
import pathlib
import ros_numpy
from sdf_tools.utils_3d import compute_sdf_and_gradient
import pickle
import rospy
import pdb
import numpy as np

from rviz_voxelgrid_visuals import conversions
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker


def visualize_sdf(pub, sdf: np.ndarray, shape, res, origin_point):
    points = get_grid_points(origin_point, res, shape)
    list_of_tuples = [(p[0], p[1], p[2], d) for p, d in zip(points.reshape([-1, 3]), sdf.flatten())]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('distance', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id='world', stamp=rospy.Time.now())
    pub.publish(msg)


def get_grid_points(origin_point, res, shape):
    indices = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.stack(indices, axis=-1)
    points = (indices * res) - origin_point
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binvox', help='the .binvox file', type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node("sdf_demo_rviz")

    vg_pub = rospy.Publisher("voxelgrid", VoxelgridStamped, queue_size=10)
    pc_pub = rospy.Publisher("points", PointCloud2, queue_size=10)
    sdf_pub = rospy.Publisher("sdf", PointCloud2, queue_size=10)
    sdf_grad_pub = rospy.Publisher("sdf_grad", MarkerArray, queue_size=10)

    rospy.sleep(0.1)
    
    with args.binvox.open("rb") as in_f:
        voxels = binvox_rw.read_as_3d_array(in_f)

    # my_vg = voxels.data
    # my_vg_translate = voxels.translate
    my_vg = np.transpose(voxels.data, [1, 0, 2])
    my_vg_translate = [voxels.translate[1], voxels.translate[0], voxels.translate[2]]
    vg = conversions.vox_to_voxelgrid_stamped(voxels.data, scale=voxels.scale, frame_id="world", origin=voxels.translate)
    sdf, sdf_grad = compute_sdf_and_gradient(my_vg, voxels.scale, my_vg_translate)

    outfilename = '../models/' + args.binvox.stem + '.pkl'
    with open(outfilename, 'wb') as out_f:
        pickle.dump(sdf, out_f)

    outfilename = '../models/' + args.binvox.stem + '_vg.pkl'
    with open(outfilename, 'wb') as out_f:
        pickle.dump(vg, out_f)

    outfilename = '../models/' + args.binvox.stem + '_grad.pkl'
    with open(outfilename, 'wb') as out_f:
        pickle.dump(sdf_grad, out_f)

    for i in range(5):
        visualize_sdf(sdf_pub, sdf, voxels.data.shape, voxels.scale, voxels.translate)
        vg_pub.publish(vg)
        rospy.sleep(0.1)

if __name__ == "__main__":
    main()
