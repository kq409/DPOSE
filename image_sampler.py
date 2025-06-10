import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

import utils

def load_data(txt_file_path):
    """Load data"""
    info = []
    with open(txt_file_path, "r") as file:
        for line in file:
            values = [float(x) for x in line.strip().split(",")]
            info.append(np.array(values))
    return np.array(info)


def get_valid_camera_positions(scene, banana_pos, num_cameras=5, min_dist=0.1, max_dist=1.5):
    """
    Sample camera positions randomly in a scene.
    Use get_random_point_by_room_type("kitchen") to make sure sampled positions are in the kitchen.

    Args:
        scene: iGibson scene instance
        banana_pos: Banana position
        num_cameras: number of cameras
        min_dist: minimum distance between the banana and sampled positions
        max_dist: maximum distance between the banana and sampled positions
    """

    valid_camera_positions = []
    attempts = 0

    while len(valid_camera_positions) < num_cameras:
        _, cam_pos = scene.get_random_point_by_room_type("kitchen")  # Sample positions
        cam_pos[2] = 1.7  # Fix camera height

        # Compute X-Y distance from camera to banana
        distance = np.linalg.norm(cam_pos[:2] - banana_pos[:2])

        # If is within range and banana is visible
        if min_dist <= distance <= max_dist and is_visible_from_camera(cam_pos.tolist(), banana_pos.tolist()):
            valid_camera_positions.append(cam_pos)

        attempts += 1

    print(f"Number of {len(valid_camera_positions)} valid camera positions found (min={min_dist}, max={max_dist})")  # Print for Debugging
    return np.array(valid_camera_positions)



def is_visible_from_camera(camera_pos, target_pos):
    """
    Use PyBullet rayTest() to check if visible
    """

    ray_test_results = p.rayTest(camera_pos, target_pos)
    hit_object_id = ray_test_results[0][0]  # Get collision boject ID

    if hit_object_id == -1:
        return True
    else:
        body_name = p.getBodyInfo(hit_object_id)[0].decode("utf-8")  # Get object name
        body_pos, _ = p.getBasePositionAndOrientation(hit_object_id)  # Get object position
        print("Blocked body: ", body_name)
        return False


def sample(selection="user", headless=False, dataset_path = "../dataset", scene_id = 'Benevolence_1_int'):
    """
    Load kitchen scene and place cameras in the scene, then take images.
    """

    save_dir = os.path.join(dataset_path, "images")
    pointcloud_dir = os.path.join(dataset_path, "pointcloud")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pointcloud_dir, exist_ok=True)


    # Load banana data
    position_file_path = os.path.join(dataset_path, "banana_positions.txt")
    rotation_file_path = os.path.join(dataset_path, "banana_rotations.txt")
    banana_positions = load_data(position_file_path)
    banana_rotations = load_data(rotation_file_path)

    # Load scene
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode="gui_interactive" if not headless else "headless",
                  image_width=256, image_height=256, rendering_settings=settings)

    scene = InteractiveIndoorScene(scene_id, build_graph=True)

    print(scene.room_ins_name_to_ins_id)
    scene_aabb_kitchen = scene.get_aabb_by_room_instance('kitchen_0')
    print(scene_aabb_kitchen)
    room_range_file = os.path.join(dataset_path, "room_range.txt")
    with open(room_range_file, "w") as room:
        room.write(f"{scene_aabb_kitchen[0][0]:.6f}, {scene_aabb_kitchen[0][1]:.6f}\n")
        room.write(f"{scene_aabb_kitchen[1][0]:.6f}, {scene_aabb_kitchen[1][1]:.6f}\n")

    s.import_scene(scene)

    # Define the banana to add
    objects_to_add = ["011_banana"]

    # Sample camera positions
    num_cameras_per_banana = 2
    num_screenshot_per_camera = 1
    valid_camera_positions = {}

    for idx in range(len(banana_positions)):
        banana_pos = np.array(banana_positions[idx])
        valid_camera_positions[idx] = get_valid_camera_positions(scene, banana_pos, num_cameras=num_cameras_per_banana)


    camera_position_file = os.path.join(dataset_path, "camera_positions.txt")
    camera_rotation_file = os.path.join(dataset_path, "camera_rotations.txt")
    banana_camera_position_file = os.path.join(dataset_path, "banana_camera_positions.txt")
    camera_position_minmax_file = os.path.join(dataset_path, "camera_position_minmax.txt")

    # Open files
    with open(camera_position_file, "w") as cam_pos_file, \
            open(camera_rotation_file, "w") as cam_rot_file, \
            open(banana_camera_position_file, "w") as banana_pos_file:

        # Add banana and take images
        scene_count = 0
        obj = YCBObject(objects_to_add[0])
        s.import_object(obj)  # Import banana
        for banana_idx, cam_positions in valid_camera_positions.items():

            banana_pos = banana_positions[banana_idx]
            banana_rpy = banana_rotations[banana_idx]

            # For iGibson, additional rotation added.
            pybullet_quat = quat_from_euler(banana_rpy)
            correction_quat = p.getQuaternionFromEuler([0, -np.pi / 2, -np.pi / 2])
            final_quat = p.multiplyTransforms([0, 0, 0], pybullet_quat, [0, 0, 0], correction_quat)[1]

            obj.set_position_orientation(tuple(banana_pos), tuple(final_quat))
            print(f"banana_pos: {banana_pos}")
            print(f"banana_rpy: {banana_rpy} quat: {final_quat}")

            # Compute camera view vector
            for cam_idx, cam_pos in enumerate(cam_positions):
                view_dir = banana_pos - np.array(cam_pos)
                view_dir /= np.linalg.norm(view_dir)

                for pose_count in range(num_screenshot_per_camera*2):
                    if pose_count == 0:
                        random_offset = np.random.uniform(-0.5, 0.5, 3)  # Random offset
                        view_dir_offset = view_dir + random_offset
                        view_dir_offset /= np.linalg.norm(view_dir_offset)  # Normalization
                        view = view_dir_offset
                    elif pose_count == 1:
                        random_view = np.random.randn(3)
                        random_view /= np.linalg.norm(random_view)
                        view = random_view

                    s.viewer.initial_pos = cam_pos.tolist()
                    s.viewer.initial_view_direction = view.tolist()
                    s.viewer.reset_viewer()

                    s.step()

                    camera_quat = utils.compute_camera_quaternion(view)
                    banana_pos_camera = utils.transform_banana_to_camera(banana_pos, cam_pos, camera_quat)

                    cam_pos_file.write(f"{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}\n")
                    cam_rot_file.write(
                        f"{view[0]:.6f}, {view[1]:.6f}, {view[2]:.6f}\n")
                    banana_pos_file.write(
                        f"{banana_pos_camera[0]:.6f}, {banana_pos_camera[1]:.6f}, {banana_pos_camera[2]:.6f}\n")

                    frames = s.renderer.render(modes=("rgb", "3d"))
                    rgb = frames[0]
                    pointcloud = frames[1][:, :, :3]


                    save_path = os.path.join(save_dir, f"scene_{scene_count:03d}_cam_{cam_idx:03d}_pose_{pose_count:03d}.png")
                    pointcloud_save_path = os.path.join(pointcloud_dir, f"scene_{scene_count:03d}_cam_{cam_idx:03d}_pose_{pose_count:03d}.npy")

                    np.save(pointcloud_save_path, pointcloud)

                    plt.imsave(save_path, rgb)
                    print(f"Image saved: {save_path}")  # Debug


            scene_count += 1

        s.disconnect()

    with open(camera_position_minmax_file, "w") as cam_pos_minmax_file:

        data = np.genfromtxt(camera_position_file, delimiter=",", dtype=float, invalid_raise=False)
        min_values = np.nanmin(data, axis=0)
        max_values = np.nanmax(data, axis=0)

        cam_pos_minmax_file.write(f"{min_values[0]:.6f}, {min_values[1]:.6f}, {min_values[2]:.6f}\n")
        cam_pos_minmax_file.write(f"{max_values[0]:.6f}, {max_values[1]:.6f}, {max_values[2]:.6f}\n")