import cv2
import os

def write_video(video_path, image_dir, width=1920, height=1080):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 1, (width, height))
    for image_name in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        video.write(image)
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    in_dir = 'overlayed'
    out_dir = 'overlayed_video'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for video_id in os.listdir(in_dir):
        if not os.path.isdir(os.path.join(in_dir, video_id)):
            continue
        video_name = video_id + '.mp4'
        video_path = os.path.join(out_dir, video_name)
        image_dir = os.path.join(in_dir, video_id)
        write_video(video_path, image_dir)

    

    

