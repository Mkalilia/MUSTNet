from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """
    Superclass for monocular dataloaders
    Args:
        data_path : raw img data path
        pseudo_depth_path : the pre-generated pseudo depth (npy)
        flow_mask : the pre-genereated optical flow for dynamic mask (npy)
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 pseudo_depth_path,
                 flow_mask,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.pseudo_depth_path = pseudo_depth_path
        self.flow_mask = flow_mask
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.load_depth = False

    def preprocess(self, inputs, color_aug):
        """
        Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:

            if self.is_train:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                if self.pseudo_depth_path is not None and i==0:
                    inputs[("depth_p", 0)] = self.get_depth_psuedo_np(folder, frame_index, side, do_flip)
                if self.flow_mask is not None and i==0:
                    inputs[("flow_forward", 0)] = self.get_flow_forward(folder, frame_index, side, do_flip)
                    inputs[("flow_backward", 0)] = self.get_flow_backward(folder, frame_index, side, do_flip)

            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                if self.pseudo_depth_path is not None and i==0:
                    inputs[("depth_p", 0)] = self.get_depth_psuedo_np(folder, frame_index, side, do_flip)
                if self.flow_mask is not None and i==0:
                    inputs[("flow_forward", 0)] = self.get_flow_forward(folder, frame_index, side, do_flip)
                    inputs[("flow_backward", 0)] = self.get_flow_backward(folder, frame_index, side, do_flip)

        """
        adjusting intrinsics to match each scale in the pyramid
        """
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    def get_depth_psuedo_np(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    def get_flow_forward(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    def get_flow_backward(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

