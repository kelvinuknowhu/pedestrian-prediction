import torch
import torch.nn as nn

from PIL import Image
import numpy as np
import math
import time
import json
import os
import cv2
import sys
import logging

import drn
import data_transforms

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None, pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.seg = nn.Conv2d(model.out_dim, classes, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def save_output_image(pred, filename, output_dir):
    """
    Saves a given (C x H x W) into an image file.
    """
    im = Image.fromarray(pred)
    fn = os.path.join(output_dir, filename + '.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im.save(fn)


def save_colorful_image(pred, filename, output_dir, palette):
    """
    Saves a given (C x H x W) into an image file.
    """
    im = Image.fromarray(palette[pred.squeeze()])
    fn = os.path.join(output_dir, filename + '.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im.save(fn)


def test(image, model, num_classes, name, output_dir='prediction', save_vis=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    data_time.update(time.time() - end)
    with torch.no_grad():
        image_var = torch.autograd.Variable(image, requires_grad=False)
        final = model(image_var)[0]
        _, pred = torch.max(final, dim=1)
        pred = pred.cpu().data.numpy()
        pred = pred.astype(np.uint8)
        pred = pred.squeeze(axis=0)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_image(pred, name, output_dir)
            save_colorful_image(pred, name, output_dir + '_color', CITYSCAPE_PALETTE)
        end = time.time()
        logger.info('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              .format(batch_time=batch_time, data_time=data_time))


def get_video_frames(video_name, frames):
    video = cv2.VideoCapture(video_name)
    i = 1
    while True:
        ret, img = video.read()
        if not ret:
            return
        if i in frames:
            yield img
        i += 1


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


if __name__ == '__main__':

    fold_dict, frames_dict, names_dict = get_dicts()
    arch = 'drn_d_38'
    classes = 19
    pretrained = 'pretrained/drn_d_38_cityscapes.pth'

    # Load the DRN model
    model = DRNSeg(arch, classes, pretrained_model=None, pretrained=True)
    state_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        new_state_dict = {}
        for key in state_dict:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict
    if torch.cuda.is_available():
        model.load_state_dict(state_dict)
    else:
        logger.warning("CUDA not available!\n")
        model.load_state_dict(state_dict)

    # Transformations that need to be performed on input image
    transforms = data_transforms.Compose([
        data_transforms.ToTensor()
    ])

    for video_id in sorted(frames_dict):
        video_name = os.path.join('clips', video_id + '.mp4')
        images = get_video_frames(video_name, frames_dict[video_id])

        for i, image in enumerate(images):
            image_name = names_dict[video_id][i]
            logger.info(image_name)
            # Swap image from (H, W, C) to (C, H, W)
            image = np.transpose(image, axes=(2, 0, 1))
            # Make batch size 1
            image = np.expand_dims(image, axis=0)
            # Turn numpy array into tensor
            image = transforms(image)[0]
            test(image, model, classes, image_name)
    