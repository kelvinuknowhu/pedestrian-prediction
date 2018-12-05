import os
import shutil


def reorganize_mask_dir(mask_dir):
    for filename in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, filename)
        dir_name = filename.split('-')[0]
        new_filename = str(int(filename.split('-')[1].split('.')[0]) - 1) + '.png'
        dir_path = os.path.join(mask_dir, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        new_file_path = os.path.join(dir_path, new_filename)
        shutil.move(file_path, new_file_path)

if if __name__ == "__main__":
    reorganize_mask_dir('/scratch/datasets/JAAD/predictions')