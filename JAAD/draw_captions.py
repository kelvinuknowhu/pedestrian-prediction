import cv2
from PIL import Image
import os
import json

CATEGORIES = {
    0: "NOT CROSS",
    1: "CROSS"
}

def get_dicts():
    fold_dict_filename = 'pedestrian_dataset_folds/fold_dict.json'
    fold_dict = json.load(open(fold_dict_filename, 'r'))
    num_frames = 30
    frames_to_process = set()
    for json_filename in fold_dict:
        json_path = os.path.join(fold_dict[json_filename], json_filename)
        ped_json = json.load(open(json_path, 'r'))
        video_name = ped_json['video']
        first_frame = ped_json['frame_data'][0]
        start = first_frame['frame_index']
        for idx in range(start, start + num_frames):
            frames_to_process.add(video_name + '-' + str(idx))

    frames_dict = {}
    for frame in frames_to_process:
        video, idx = frame.split('-')
        if video not in frames_dict:
            frames_dict[video] = []
        frames_dict[video].append(int(idx))
    for video in frames_dict:
        frames_dict[video] = sorted(frames_dict[video])

    names_dict = {}
    for video in frames_dict:
        if video not in names_dict:
            names_dict[video] = []
        for idx in frames_dict[video]:
            names_dict[video].append(video + '-' + str(idx))

    return fold_dict, frames_dict, names_dict

# predictions/video_0001-2.json: All predictions in frame index 2 of video_0001
def overlay_predictions(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions: the result of the computation by the model.
            It should contain the field `labels` and `bbox`.
    """
    labels = predictions['labels']
    labels = [CATEGORIES[label] for label in labels]
    boxes = predictions['bbox']

    template = "{}"
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        s = template.format(label)
        cv2.rectangle(
            image, (x1, y1) ,(x2, y2), (0,255,0), 2
        )
        cv2.putText(
            image, s, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return image

def get_video_frames(video_name, frames):
    video = cv2.VideoCapture(video_name)
    i = 0
    while True:
        ret, img = video.read()
        if not ret:
            return
        if i in frames:
            yield img
        i += 1

if __name__ == '__main__':

    fold_dict, frames_dict, names_dict = get_dicts()    
    clips_dir = '/scratch/datasets/JAAD_clips'
    overlayed_dir = '/scratch/datasets/JAAD/overlayed'
    # clips_dir = 'clips'
    # overlayed_dir = 'overlayed'

    for video_id in frames_dict:
        print(video_id)
        video_name = os.path.join(clips_dir, video_id + '.mp4')
        frames = frames_dict[video_id]
        images = get_video_frames(video_name, frames)

        with open('ground_truth_dict.json', 'r') as f:
            ground_truth_dict = json.load(f)
        

        predictions_dict = ground_truth_dict[video_id]

        for i, image in enumerate(images):
            frame = frames[i]
            overlayed_image = overlay_predictions(image, predictions_dict[str(frame)])
            out_dir = os.path.join(overlayed_dir, video_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            overlayed_path = os.path.join(out_dir, str(frame) + '.png')        
            Image.fromarray(overlayed_image).save(overlayed_path)

