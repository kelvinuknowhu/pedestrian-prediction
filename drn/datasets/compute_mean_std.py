import argparse
import json
import numpy as np
from PIL import Image
from os import path


def compute_mean_std(data_dir):
    image_list_path = path.join(data_dir, 'train_images.txt')
    image_list = [line.strip() for line in open(image_list_path, 'r')]
    np.random.shuffle(image_list)
    pixels = []
    for i, image_pathin in enumerate(image_list[:500]):
        print("[i+1/500] {}".format(image_path))
        image = Image.open(path.join(data_dir, image_path), 'r')
        pixels.append(np.asarray(image).reshape(-1, 3))
    pixels = np.vstack(pixels)
    mean = np.mean(pixels, axis=0) / 255
    std = np.std(pixels, axis=0) / 255
    print(mean, std)
    info = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(path.join(data_dir, 'info.json'), 'w') as fp:
        json.dump(info, fp)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute mean and std of a dataset.')
    parser.add_argument('--data-dir', default='./', required=True,
                        help='Need to specify the data folder where train_images.txt resides.')
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    compute_mean_std(args.data_dir)

if __name__ == '__main__':
    main()
