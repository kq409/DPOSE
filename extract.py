import os
import xml.etree.ElementTree as ET
import numpy as np

def extract_urdf(num_files, folder_path, dataset_path = "../dataset"):
    """
    Extract banana positions from URDF files.


    :param folder_path: Folder path for URDF files
    :param dataset_path: Folder path for dataset
    :param num_files: Number of URDF files
    """

    # URDF path
    # urdf_folder = r'D:\Research\Life_Long_Learning\diffusion_proejct\diffusion\diffusion_behavior\code\banana_data\new_dataset'

    # Output path
    xyz_output_file = os.path.join(dataset_path, 'banana_positions.txt')
    rpy_output_file = os.path.join(dataset_path, 'banana_rotations.txt')
    minmax_output_file = os.path.join(dataset_path, 'banana_positions_minmax.txt')

    # Init
    xyz_data = []
    rpy_data = []

    # Traverse num_files urdf file
    for i in range(num_files):
        urdf_path = os.path.join(folder_path, f'Benevolence_1_int_task_place_banana_0_{i}.urdf')

        if not os.path.isfile(urdf_path):
            print(f"Warning: {urdf_path} not found, skipping...")
            continue

        # Extraction
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('link'):
            if 'banana' in link.get('name', ''):
                xyz_str = link.get('xyz', '0 0 0')
                object_positions = np.array([float(x) for x in xyz_str.split()])

                rpy_str = link.get('rpy', '0 0 0')
                object_rotations = np.array([float(x) for x in rpy_str.split()])

                xyz_data.append(object_positions)
                rpy_data.append(object_rotations)

                print(f"Extracted from {urdf_path}:")
                print(f"Banana Position: {object_positions}")
                print(f"Banana Rotation (RPY): {object_rotations}")
                break

    # Store data as .txt file

    min_values = np.nanmin(np.array(xyz_data), axis=0)  # Ignore NaN
    max_values = np.nanmax(np.array(xyz_data), axis=0)

    with open(minmax_output_file, "w") as m:
        m.write(f"{min_values[0]:.6f}, {min_values[1]:.6f}, {min_values[2]:.6f}\n")
        m.write(f"{max_values[0]:.6f}, {max_values[1]:.6f}, {max_values[2]:.6f}\n")

    np.savetxt(xyz_output_file, np.array(xyz_data), fmt="%.6f", delimiter=",")
    np.savetxt(rpy_output_file, np.array(rpy_data), fmt="%.6f", delimiter=",")


    print(f"Saved {len(xyz_data)} banana positions to {xyz_output_file}")
    print(f"Saved {len(rpy_data)} banana rotations to {rpy_output_file}")
