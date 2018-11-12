import re
import cv2
import numpy as np
import argparse
import os
import math
import random

import keras

from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalMaxPool2D
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.utils import Sequence

from sklearn.metrics import average_precision_score

IMAGE_DIMS = (1920,1080)

#Prepare video frames with traffic_scene_elements.txt to produce data pairs for classifier

class SceneDetector:
    def __init__(self, model):
        self.model = model

    def train(self, xTr, yTr):
        self.model.compile(
            optimizer=SGD(lr=0.01, momentum = 0.9, decay=0.0005),
            loss=binary_crossentropy,
            metrics=["accuracy"]
            )
        self.model.fit(xTr, yTr, epochs=5, batch_size=32)

    def train_gen(self, traingen, testgen):
        self.model.compile(
            optimizer=SGD(lr=0.01, momentum = 0.9, decay=0.0005),
            loss=binary_crossentropy,
            metrics=["accuracy"]
            )
        self.model.fit_generator(traingen, epochs=5, validation_data=testgen)

    def classify(self, x):
        return self.model.predict(x)

    def save_weights(self, file):
        self.model.save_weights(file)

    def load_weights(self, file):
        self.model.load_weights(file)


class Dataset(Sequence):
    def __init__(self, clip_names, num_frames, batch_size, clip_dir, label_dir):
        self.clip_names = clip_names
        self.lengths = num_frames
        self.batch_size = batch_size
        self.clip_dir = clip_dir
        self.label_dir = label_dir

        self.idx_clips = []
        for clip, num in zip(clip_names, num_frames):
            self.idx_clips += [(clip, i) for i in range(num)]

        self.ys = np.concatenate([
            np.load("{}/{}.npy".format(self.label_dir, clip)) \
            for clip in self.clip_names \
            ])
        print(len(clip_names), len(num_frames), clip_names[0], num_frames[0])
        print("Data loaded: {} images, {} labels".format(len(self.idx_clips), self.ys.shape[0]))

    def __len__(self):
        return math.ceil(self.ys.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size \
            if (idx+1)*self.batch_size < self.ys.shape[0] else self.ys.shape[0]

        x = np.array([
            cv2.imread("{}/{}/{:04}.jpg".format(self.label_dir, clip, num)) \
            for clip, num in self.idx_clips[start : end] \
            ])

        y = self.ys[start : end]
        print(self.idx_clips[start : end])
        return x, y



def get_frames(vidfile):
    vidcap = cv2.VideoCapture(vidfile)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    print(frame_count)
    vidcap = cv2.VideoCapture(vidfile)

    if IMAGE_DIMS != None:
        frame_width, frame_height = IMAGE_DIMS

    print("Video dims: ({},{}) - {}".format(frame_width, frame_height, vidfile))
    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    fc = 0
    ret = True


    while (fc < frame_count and ret):
        ret, img = vidcap.read()
        cv2.resize(img, (frame_width, frame_height), buf[fc])

        fc += 1

    vidcap.release()
    return buf


def prepare_data(tagfile, vid_directory, data_directory):
    #Read traffic_scene_elements.txt, split by lines
    lines = []

    clip_names = []
    num_frames = []

    with open(tagfile) as f:
        lines = f.readlines()[1:]

    if not os.path.isdir(data_directory):
        print("Creating directory...")

        os.mkdir(data_directory)
        for line in lines:
            vid_frames = line.split(',', 1)

            vid_name = vid_frames[0]
            vid_attribs = re.findall(r"[a-zA-Z0-9_][^;]*", vid_frames[1])

            print("Loading: {}".format(vid_name))

            #1-stop_sign 2-traffic_light 3-red_light 4-green_light 5-ped_sign 6-ped_crossing 7-parking_lot, 8-Garage

            #For video file frames, set output vector based on attribute presence
            #For elements in vid_attribs...
            #  If number followed by nothing, set (n-1)th output entry of all frames to 1
            #  If number followed by range, set (n-1)th output entry of frames in range (inclusive) to 1

            frames = get_frames("{}/{}.mp4".format(vid_directory, vid_name))
        
            labels = np.zeros((frames.shape[0], 8))  
            for it in vid_attribs:
                if len(it) == 1:
                    labels[:,int(it)-1] = 1
                else:
                    #Handle multiple ranges
                    nums = re.findall(r"[0-9]+", it)
                    entry = int(nums[0]) - 1
                    for i in range(1,len(nums),2):   
                        begin = int(nums[i]) - 1
                        end = int(nums[i+1]) - 1

                        labels[begin:end + 1, entry] = 1

            clip_names += [vid_name]
            num_frames += [frames.shape[0]]
            #Write data directory
            new_data_path = "{}/{}".format(data_directory,vid_name)

            os.mkdir(new_data_path)
            for i in range(frames.shape[0]):
                cv2.imwrite("{}/{:04}.jpg".format(new_data_path, i), frames[i])

            np.save(new_data_path, labels)
    else:
        for clip in sorted(filter(lambda x: len(x.split("."))==2, os.listdir(data_directory))):
            clip_names += [clip.split(".")[0]]
            num_frames += [np.load("{}/{}".format(data_directory, clip)).shape[0]]

    return clip_names, num_frames

def build_model():
    return Sequential([
        Conv2D(96, 11, strides=4, padding='valid', activation='relu',
                input_shape=(None, None, 3)),
        MaxPooling2D(pool_size=3, strides=2),
        Conv2D(256, 5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(256, 3, padding='same', activation='relu'),
        GlobalMaxPool2D(),
        Dense(8)
    ])

def build_model_resnet():
    model = ResNet50(
        include_top=False, 
        weights="imagenet", 
        input_shape=(None, None, 3), 
        pooling="max"
        )
    model.add(Dense(8))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save",
        nargs='?',
        const="detector_weights.h5",
        help="Path to a file to save weights")

    parser.add_argument("-m", "--model",
        default="alexnet",
        help="Type of model")

    parser.add_argument("clips",
        help="Path to a directory containing JAAD clips")
    parser.add_argument("tagfile",
        help="Path to a file that contains JAAD scene tag.")
    parser.add_argument("dataset",
        help="Path to a directory that contains processed JAAD data")
    parser.add_argument("-l", "--load", help="Path to saved weights")

    args = parser.parse_args()

    save_dest = args.save

    videopath = args.clips
    tagpath = args.tagfile
    datapath = args.dataset
    modeltype = args.model

    if modeltype == "resnet50":
        model = build_model_resnet()
    else:
        model = build_model()


    if args.load != None:
        names, frames = prepare_data(tagpath, videopath, datapath)

        detector = SceneDetector(model)
        detector.load_weights(args.load)
        dataset = Dataset(names, frames, 32, videopath, datapath)

        total_acc = 0
        ys = []
        y_preds = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            y_raw = detector.classify(x)
            y_pred = np.heaviside(y_raw, 0)
            binary_acc = np.average(np.logical_not(np.logical_xor(y, y_pred)), axis=0)
            total_acc = total_acc + binary_acc
            ys += [y]
            y_preds += [y_raw]

        ys = np.concatenate(ys)
        y_preds = np.concatenate(y_preds)
        
        aps = [average_precision_score(ys[:,i], y_preds[:,i]) for i in range(8)]
        total_ap = average_precision_score(ys, y_preds)

        total_acc /= len(dataset)
        print("Total test accuracy: {}".format(total_acc))
        print("Average precisions: {}".format(aps))
        print("Total Average precision: {}".format(total_ap))


    else:
        names, frames = prepare_data(tagpath, videopath, datapath)

        print("Creating training sets...")
        randidx = np.random.permutation(len(names))
        names_shuff = [names[i] for i in randidx]
        frames_shuff = [frames[i] for i in randidx]

        dataset_train = Dataset(names_shuff[:len(names)*4//5], frames_shuff[:len(names)*4//5], 32, videopath, datapath)
        dataset_test = Dataset(names_shuff[len(names)*4//5:], frames_shuff[len(names)*4//5:], 32, videopath, datapath)

        detector = SceneDetector(model)
        
        print("Training...")
        detector.train_gen(dataset_train, dataset_test)

        if save_dest != None:
            detector.save_weights(save_dest)


        total_acc = 0
        ys = []
        y_preds = []
        for i in range(len(dataset_test)):
            x, y = dataset_test[i]
            y_raw = detector.classify(x)
            y_pred = np.heaviside(y_raw, 0)
            binary_acc = np.average(np.logical_not(np.logical_xor(y, y_pred)), axis=0)
            total_acc = total_acc + binary_acc
            ys += [y]
            y_preds += [y_raw]

        ys = np.concatenate(ys)
        y_preds = np.concatenate(y_preds)
        
        aps = [average_precision_score(ys[:,i], y_preds[:,i]) for i in range(8)]
        total_ap = average_precision_score(ys, y_preds)

        total_acc /= len(dataset_test)
        print("Total test accuracy: {}".format(total_acc))
        print("Average precisions: {}".format(aps))
        print("Total Average precision: {}".format(total_ap))

        

    

