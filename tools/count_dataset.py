import os
import h5py

# process bar
from tqdm import tqdm


def get_frame_num(hdf_file):
    with h5py.File(hdf_file, "r") as f:
        return f["timestamp"][()].shape[0]


def count_dataset(dataset_path):
    count = 0
    total_frame_num = 0
    for filename in os.listdir(dataset_path):
        if filename.endswith(".hdf5"):
            count += 1
            cur_frame_num = get_frame_num(os.path.join(dataset_path, filename))
            total_frame_num += cur_frame_num
    return count, total_frame_num


DATASETS = {
    "pick and place": [
        "grab_to_plate2_v1",
        "grab_to_plate2",
        "grab_to_plate1_v1",
        "grab_to_plate1",
        "grab_pencil_v1",
        "grab_pencil2_v1",
        "grab_pencil1_v1",
        "grab_cup_v1",
        "grab_cup",
        "grab_cube2_v1",
        "grab_cube",
    ],
    "push and pull": [
        "push_plate_v1",
        "push_plate",
        "push_box_two_v1",
        "push_box_random_v1",
        "push_box_common_v1",
        "push_box",
        "pull_plate_v1",
        "pull_plate",
    ],
    "writing": [
        "writing",
        "writing_a",
        "writing_circle",
        "writing_circle_inv",
        "writing_cross",
        "writing_random",
        "writing_rec",
        "writing_t",
        "writing_tri",
    ],
    "rolling": ["roll"],
    "H&R-S": [
        "roll",
        "writing",
        "writing_a",
        "writing_circle",
        "writing_circle_inv",
        "writing_cross",
        "writing_random",
        "writing_rec",
        "writing_t",
        "writing_tri",
        "push_plate_v1",
        "push_plate",
        "push_box_two_v1",
        "push_box_random_v1",
        "push_box_common_v1",
        "push_box",
        "pull_plate_v1",
        "pull_plate",
        "grab_to_plate2_v1",
        "grab_to_plate2",
        "grab_to_plate1_v1",
        "grab_to_plate1",
        "grab_pencil_v1",
        "grab_pencil2_v1",
        "grab_pencil1_v1",
        "grab_cup_v1",
        "grab_cup",
        "grab_cube2_v1",
        "grab_cube",
    ],
    "H&R-L": [
        "pull_plate_grab_cube",
        "grab_two_cubes2_v1",
        "grab_two_cubes2",
        "grab_two_cubes1",
        "grab_to_plate2_and_pull_v1",
        "grab_to_plate2_and_back_v1",
        "grab_to_plate1_and_back_v1",
        "grab_both_cubes_v1",
    ],
}

for dataset_name, dataset_list in DATASETS.items():
    print(f"Dataset: {dataset_name}")
    total_count = 0
    total_frame_num = 0
    for dataset in tqdm(dataset_list, desc="Processing"):
        dataset_path = f"/data1/dataset/{dataset}"
        count, frame_num = count_dataset(dataset_path)
        total_count += count
        total_frame_num += frame_num

    print(
        f"Dataset: {dataset_name}, Count: {total_count}, Total Frame Number: {total_frame_num}"
    )
    print()
