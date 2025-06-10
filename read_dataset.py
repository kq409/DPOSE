import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import json
import numpy as np

import torch

class CoordMinMaxNormalize(object):
    def __init__(self, min_val, max_val):
        self.min_val = torch.tensor(min_val, dtype=torch.float32).view(3)
        self.max_val = torch.tensor(max_val, dtype=torch.float32).view(3)

    def __call__(self, coord):
        normalized_coord = (coord - self.min_val) / (self.max_val - self.min_val)
        normalized_coord = torch.clamp(normalized_coord, 0, 1)
        return normalized_coord.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + '(3D Coord MinMax Normalization)'



class BananaDataset(Dataset):
    def __init__(self, h5_path, json_path, transform_RGB=None, transform_Gray=None,
                 transform_depth=None):
        self.h5_path = h5_path
        self.transform_RGB = transform_RGB
        self.transform_Gray = transform_Gray
        self.transform_depth = transform_depth

        # Read JSON
        with open(json_path, "r") as f:
            self.metadata = json.load(f)

        self.keys = [k for k in self.metadata.keys() if k != "floor_plan"]

        self.floor_plan_path = self.metadata["floor_plan"]
        self.floor_plan = Image.open(self.floor_plan_path).convert("L")

        self.banana_positions_minmax = self.metadata["banana_positions_minmax"]
        self.room_range = self.metadata["room_range"]
        self.camera_positions_minmax = self.metadata["camera_positions_minmax"]

        self.x_min = self.room_range[0][0]
        self.y_min = self.room_range[0][1]
        self.z_min = 0
        self.x_max = self.room_range[1][0]
        self.y_max = self.room_range[1][1]
        self.z_max = 3                  # Differ from house to house, should be upgraded later.
        self.x_min_camera = self.camera_positions_minmax[0][0]
        self.y_min_camera = self.camera_positions_minmax[0][1]
        self.z_min_camera = self.camera_positions_minmax[0][2]
        self.x_max_camera = self.camera_positions_minmax[1][0]
        self.y_max_camera = self.camera_positions_minmax[1][1]
        self.z_max_camera = self.camera_positions_minmax[1][2]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as self.h5_file:
            image_key = self.keys[idx]
            image_info = self.metadata[image_key]
            hdf5_index = image_info["hdf5_index"]
            image_index = self.h5_file["image_indices"][hdf5_index][3]  #[scene_id, point_id, pose_id, image_index]
            scene_index = self.h5_file["image_indices"][hdf5_index][0]
            point_index = self.h5_file["image_indices"][hdf5_index][1]
            pose_index = self.h5_file["image_indices"][hdf5_index][2]

            if pose_index < 0.5:
                visible_mask = True
            elif pose_index > 0.5:
                visible_mask = False

            # Read banana positions (world coordinate)
            banana_position = torch.tensor(self.h5_file["banana_positions"][scene_index])
            banana_rotation = torch.tensor(self.h5_file["banana_rotations"][scene_index])


            # Read banana positions (camera coordinate)
            banana_camera_positions = torch.tensor(self.h5_file["banana_camera_positions"][image_index])

            # Read camera positions and rotations
            camera_position = torch.tensor(self.h5_file["camera_positions"][image_index])
            camera_rotation = torch.tensor(self.h5_file["camera_rotations"][image_index])

            # Read RGB images and depth data
            img_path = image_info["image_path"]
            image = Image.open(img_path).convert("RGB")

            depth_path = image_info["depth_path"]
            depth = Image.open(depth_path).convert("L")
            # depth = torch.tensor(np.load(depth_path), dtype=torch.float32).unsqueeze(0)


            if self.transform_RGB:
                image = self.transform_RGB(image)

            if self.transform_Gray:
                floor_plan = self.transform_Gray(self.floor_plan)


            banana_position_normalized = CoordMinMaxNormalize(min_val=[self.x_min, self.y_min, self.z_min],
                                                    max_val=[self.x_max, self.y_max, self.z_max])(banana_position)  # To be upgraded

            banana_rotation_normalized = CoordMinMaxNormalize(min_val=[self.x_min, self.y_min, self.z_min],
                                                              max_val=[self.x_max, self.y_max, self.z_max])(banana_rotation)    # To be upgraded

            banana_camera_positions_normalized = CoordMinMaxNormalize(min_val=[self.x_min_camera, self.y_min_camera, self.z_min_camera],
                                                              max_val=[self.x_max_camera, self.y_max_camera, self.z_max_camera])(
                banana_camera_positions)
            camera_position_normalized = CoordMinMaxNormalize(min_val=[self.x_min, self.y_min, self.z_min],
                                                                      max_val=[self.x_max, self.y_max, self.z_max])(
                camera_position)
            camera_rotation_normalized = camera_rotation


            if self.transform_depth:
                depth = self.transform_depth(depth)


        return {
            "image": image,
            "floor_plan": floor_plan,
            "banana_position": banana_position_normalized,
            "banana_rotation": banana_rotation_normalized,
            "banana_camera_positions": banana_camera_positions_normalized,
            "camera_position": camera_position_normalized,
            "camera_rotation": camera_rotation_normalized,
            "depth": depth,
            "scene_index": scene_index,
            "point_index": point_index,
            "pose_index": pose_index,
            "visible_mask": visible_mask
        }