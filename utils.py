import numpy as np
import pybullet as p
import igibson
from igibson.external.pybullet_tools.utils import quat_from_matrix, matrix_from_quat

def compute_camera_quaternion(view_dir):
    """
    Transform the camera direction vector 'view_dir' into quaternions in iGibson.

    :param view_dir: Camera direction vector
    :return: iGibson quaternions (xyzw)
    """
    view_dir = np.array(view_dir) / np.linalg.norm(view_dir)  # Normalization

    # Set "up" (Y axis), usually [0, 0, 1]
    up = np.array([0, 0, 1])

    # Set "right" (X axis)，make sure orthogonal
    right = np.cross(up, view_dir)
    right /= np.linalg.norm(right)

    # Recomputer "up" (Y axis)，make sure orthogonal
    up = np.cross(view_dir, right)

    # Make rotation matrix (3*3)
    rotation_matrix = np.column_stack((right, up, -view_dir))

    # Transform into quaternions
    pybullet_quat = quat_from_matrix(rotation_matrix)   # xyzw

    # Extra correction for iGibson
    correction_quat = p.getQuaternionFromEuler([0, -np.pi / 2, -np.pi / 2])  # iGibson
    final_quat = p.multiplyTransforms([0, 0, 0], pybullet_quat, [0, 0, 0], correction_quat)[1]

    return tuple(final_quat)


def transform_banana_to_world(camera_pos, camera_quat, banana_pos_cam):
    """
    Compute banana position in world coordinate (ignore rotation)

    :param camera_pos: Camera position in world coordinate (x_c, y_c, z_c)
    :param camera_quat: Camera rotation (qx, qy, qz, qw)
    :param banana_pos_cam: Banana position in camera coordinate (x_b', y_b', z_b')
    :return: Banana position in world coordinate (x_b, y_b, z_b)
    """

    # Transform quaternions into rotation matrix
    R_c = matrix_from_quat(camera_quat)


    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_c
    T_world_cam[:3, 3] = camera_pos

    # Compute banana homogeneous coordinates
    banana_pos_cam_h = np.append(banana_pos_cam, 1)  # (x, y, z, 1)

    # Compute banana position in world coordinate
    banana_pos_world_h = T_world_cam @ banana_pos_cam_h  # 4x4 * 4x1

    return banana_pos_world_h[:3]


def transform_banana_to_camera(banana_world, camera_pos, camera_quat):
    """
    Compute banana position in camera coordinate

    :param banana_world: Banana position in world coordinate (x_b, y_b, z_b)
    :param camera_pos: Camera position in world coordinate (x_c, y_c, z_c)
    :param camera_quat: Camera rotation (qx, qy, qz, qw)
    :return: Banana position in camera coordinate (x_b', y_b', z_b')
    """

    # Transform quaternions into rotation matrix
    R_cam = matrix_from_quat(camera_quat)

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_cam.T
    T_world_cam[:3, 3] = -R_cam.T @ camera_pos

    # Compute banana homogeneous coordinates
    banana_world_h = np.append(banana_world, 1)  # (x, y, z, 1)

    # Compute banana position in camera coordinate
    banana_cam_h = T_world_cam @ banana_world_h  # 4x4 * 4x1

    return banana_cam_h[:3]
