import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_global_min_max(data_folder):
    global_min = np.inf
    global_max = -np.inf

    file_list = os.listdir(data_folder)

    for file in file_list:
        depth_img = np.load(os.path.join(data_folder, file))

        depth = np.linalg.norm(depth_img, axis=2)

        min_depth = np.min(depth)
        max_depth = np.max(depth)

        if min_depth < global_min:
            global_min = min_depth
        if max_depth > global_max:
            global_max = max_depth

    print(f"Global Min: {global_min}, Global Max: {global_max}")
    return global_min, global_max

def make_depth(pointcloud_folder = "..\dataset\pointcloud", depth_folder = "..\dataset\depth"):

    global_min, global_max = calculate_global_min_max(pointcloud_folder)

    file_list = os.listdir(pointcloud_folder)
    for file in file_list:
        depth_img = np.load(os.path.join(pointcloud_folder, file))

        depth = np.linalg.norm(depth_img, axis=2)

        normalized_depth = (depth - global_min) / (global_max - global_min)

        save_path = os.path.join(depth_folder, os.path.splitext(file)[0] + ".png")

        # np.save(save_path, normalized_depth)
        plt.imsave(save_path, normalized_depth, cmap="gray")
        print(f"Saved: {save_path}")