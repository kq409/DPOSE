import h5py
import numpy as np
import json
import os


def load_data(txt_file_path):
    info = []
    with open(txt_file_path, "r") as file:
        for line in file:
            values = [float(x) for x in line.strip().split(",")]
            info.append(np.array(values))
    return np.array(info)

def create_dataset(dataset_path = "../dataset", num_scenes = 5000, num_points_per_scene = 2, num_poses_per_point = 2):

    num_images = num_scenes * num_points_per_scene * num_poses_per_point

    position_file_path = os.path.join(dataset_path, "banana_positions.txt")
    rotation_file_path = os.path.join(dataset_path, "banana_rotations.txt")
    camera_position_file_path = os.path.join(dataset_path, "camera_positions.txt")
    camera_rotation_file_path = os.path.join(dataset_path, "camera_rotations.txt")
    banana_camera_position_file_path = os.path.join(dataset_path, "banana_camera_positions.txt")
    banana_positions_minmax_file_path = os.path.join(dataset_path, "banana_positions_minmax.txt")
    room_range_file_path = os.path.join(dataset_path, "room_range.txt")
    camera_position_minmax_file_path = os.path.join(dataset_path, "camera_position_minmax.txt")

    banana_positions = load_data(position_file_path)
    banana_rotations = load_data(rotation_file_path)
    camera_positions = load_data(camera_position_file_path)
    camera_rotations = load_data(camera_rotation_file_path)
    banana_camera_positions = load_data(banana_camera_position_file_path)
    banana_positions_minmax = load_data(banana_positions_minmax_file_path)
    room_range = load_data(room_range_file_path)
    camera_positions_minmax = load_data(camera_position_minmax_file_path)

    image_indices = []
    image_index = 0

    for scene_id in range(num_scenes):
        for point_id in range(num_points_per_scene):
            for pose_id in range(num_poses_per_point):
                image_indices.append([scene_id, point_id, pose_id, image_index])
                image_index += 1

    image_indices = np.array(image_indices)

    h5_file_path = os.path.join(dataset_path, "dataset.h5")
    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("banana_positions", data=banana_positions)
        f.create_dataset("banana_rotations", data=banana_rotations)
        f.create_dataset("camera_positions", data=camera_positions)
        f.create_dataset("camera_rotations", data=camera_rotations)
        f.create_dataset("image_indices", data=image_indices)
        f.create_dataset("banana_camera_positions", data=banana_camera_positions)

    print("HDF5 data stored！")

    metadata = {
        "floor_plan": os.path.join(dataset_path, "floor_0.png"),
        "banana_positions_minmax": banana_positions_minmax,
        "room_range": room_range,
        "camera_positions_minmax": camera_positions_minmax
    }
    image_id = 0
    json_file_path = os.path.join(dataset_path, "metadata.json")

    for scene_id in range(num_scenes):
        for point_id in range(num_points_per_scene):
            for pose_id in range(num_poses_per_point):
                image_key = f"scene_{scene_id:04d}_cam_{point_id:04d}_pose_{pose_id:04d}"
                metadata[image_key] = {
                    "image_path": os.path.join(dataset_path, "images", f"{image_key}.png"),
                    "depth_path": os.path.join(dataset_path, "depth", f"{image_key}.png"),
                    "hdf5_index": image_id,
                    "floor_plan": metadata["floor_plan"]
                }
                image_id += 1

    with open(json_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("JSON data stored")
