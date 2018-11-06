import os
import cv2
import json
import numpy as np
import png
from scipy import ndimage
import sys
import collections
from shutil import copyfile

mode = 'val'
label_file = './bdd100k_labels_images_' + mode + '.json'
output_color_dir = './color_labels/crosswalk_' + mode
output_id_dir = './labels/crosswalk_' + mode 
input_dir = './images/100k/' + mode


f = open(label_file, 'r')
labels_json = json.load(f)
f.close()

files_dict = collections.defaultdict(list)
for label_dict in labels_json:
    file_name = label_dict['name']
    label_list = label_dict['labels']
    for label in label_list:
        if label['category'] == 'lane' and label['attributes']['laneType'] == 'crosswalk':
            files_dict[file_name].extend(label['poly2d'])
del labels_json

# See how many crosswalk lines each image has
# for key, val in files_dict.items():
#     print(key, len(val))

# See the number of images that have crosswalks
# print(len(files_dict))

# Copy the original images
# for file_name in files_dict:
#     copyfile(input_dir + '/' + file_name, output_color_dir + '/' + file_name)

width = 1280
height = 720
png_writer = png.Writer(size=(width, height))
crosswalk_color = (255, 255, 0)
id_color = (0, 0, 0)
line_color = (255, 0, 0)

for file_name in files_dict:
    lines = files_dict[file_name]
    num_lines = len(lines)
    if num_lines % 2 == 1:
        continue
    print(file_name, num_lines)
    img_color = np.zeros((height, width, 3), dtype=np.uint8)
    img_id = np.zeros((height, width, 3), dtype=np.uint8)
    img_id[:] = 255
    img = ndimage.imread(input_dir + '/' + file_name)
    for  i in range(0, num_lines, 2):
        line_1 = lines[i]
        line_2 = lines[i + 1]
        poly_points = np.array(line_1['vertices'] + list(reversed(line_2['vertices'])), dtype=np.int32)
        img_color = cv2.fillPoly(img_color, [poly_points], crosswalk_color)
        img_id = cv2.fillPoly(img_id, [poly_points], id_color)
        file_id = file_name.split('.')[0]
        file_name_color = file_id + '_train_color' + '.png'
        file_name_id = file_id + '_train_id' + '.png'
        file_name_original = file_id + '.png'

        # Draw lines in the original images
        line_1_points = np.array(line_1['vertices'], dtype=np.int32)
        line_2_points = np.array(line_2['vertices'], dtype=np.int32)
        img = cv2.polylines(img, [line_1_points], False, line_color)
        img = cv2.polylines(img, [line_2_points], False, line_color)
        png_file_original = open(output_color_dir + '/' + file_name_original, 'wb')
        png_writer.write(png_file_original, np.reshape(img, (-1, width * 3)))
        png_file_original.close()

        # Write the color mask
        png_file_color = open(output_color_dir + '/' + file_name_color, 'wb')
        png_writer.write(png_file_color, np.reshape(img_color, (-1, width * 3)))
        png_file_color.close()

        # Write the id mask
        png_file_id = open(output_color_dir + '/' + file_name_id, 'wb')
        png_writer.write(png_file_id, np.reshape(img_id, (-1, width * 3)))
        png_file_id.close()
