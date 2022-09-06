import os

import numpy as np
import torch
import torch.nn as nn

import natsort

import cv2
import random

# Data Loader
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):#fdir, pdir, sdir, transform=None):
    self.rock_dir = os.path.join(data_dir,'rock/')
    self.paper_dir = os.path.join(data_dir,'paper/')
    self.scissors_dir = os.path.join(data_dir,'scissors/')

    self.transform = transform

    lst_rock = os.listdir(self.rock_dir)
    lst_paper = os.listdir(self.paper_dir)
    lst_scissors = os.listdir(self.scissors_dir)

    lst_rock = [f for f in lst_rock if f.endswith(".png") or f.endswith(".jpg")]
    lst_paper = [f for f in lst_paper if f.endswith(".png") or f.endswith(".jpg")]
    lst_scissors = [f for f in lst_scissors if f.endswith(".png") or f.endswith(".jpg")]

    self.lst_dir = [self.rock_dir] * len(lst_rock) + [self.paper_dir] * len(lst_paper) + [self.scissors_dir] * len(lst_scissors)
    self.lst_prs = natsort.natsorted(lst_rock) + natsort.natsorted(lst_paper) + natsort.natsorted(lst_scissors)
 
  def __len__(self):
    return len(self.lst_prs)

  def __getitem__(self, index): 
    self.img_dir = self.lst_dir[index]
    self.img_name = self.lst_prs[index]

    return [self.img_dir, self.img_name]

  def custom_collate_fn_train(self, data):
    size = 220
    inputImages = []
    outputVectors = []

    for sample in data:
      prs_img = cv2.imread(os.path.join(sample[0] + sample[1]))

      # Augmentation
      height, width, channel = prs_img.shape

      img_list = []
      for angle in range(10):
        # Rotate
        rotate_angle = random.uniform(20, 70)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotate_angle, 1)
        rotated_img = cv2.warpAffine(prs_img, matrix, (width, height))

        # Preprocessing
        hsvim = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype="uint8")
        upper = np.array([255, 173, 127], dtype="uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2, 2))
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, (size, size), interpolation=cv2.INTER_CUBIC)

        gaussian = np.random.normal(0, 10, (size, size))
        thresh = thresh + gaussian
        cv2.normalize(thresh, thresh, 0, 255, cv2.NORM_MINMAX, dtype=-1)

        # Flip
        flipped_img = cv2.flip(thresh, 1)[:, :, np.newaxis]
        thresh = thresh[:, :, np.newaxis]

        img_list.extend([thresh, flipped_img])

      inputImages.extend(img_list)

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.extend([np.array(1) for _ in range(20)])
      elif dir_split[-2] == 'paper':
        outputVectors.extend([np.array(0) for _ in range(20)])
      elif dir_split[-2] == 'scissors':
        outputVectors.extend([np.array(2) for _ in range(20)])

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.append(np.array(1))
      elif dir_split[-2] == 'paper':
        outputVectors.append(np.array(0))
      elif dir_split[-2] == 'scissors':
        outputVectors.append(np.array(2))

    data = {'input': inputImages, 'label': outputVectors}

    if self.transform:
      data = self.transform(data)

    return data

  def custom_collate_fn_validation(self, data):
    size = 220
    inputImages = []
    outputVectors = []

    for sample in data:
      prs_img = cv2.imread(os.path.join(sample[0] + sample[1]))

      # Preprocessing
      hsvim = cv2.cvtColor(prs_img, cv2.COLOR_BGR2YCrCb)
      lower = np.array([0, 133, 77], dtype="uint8")
      upper = np.array([255, 173, 127], dtype="uint8")
      skinRegionHSV = cv2.inRange(hsvim, lower, upper)
      blurred = cv2.blur(skinRegionHSV, (2, 2))
      ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

      thresh = cv2.resize(thresh, (size, size), interpolation=cv2.INTER_CUBIC)

      gaussian = np.random.normal(0, 10, (size, size))
      thresh = thresh + gaussian
      cv2.normalize(thresh, thresh, 0, 255, cv2.NORM_MINMAX, dtype=-1)

      gray_img = thresh
      gray_img = gray_img[:, :, np.newaxis]

      inputImages.append(gray_img)

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.append(np.array(1))
      elif dir_split[-2] == 'paper':
        outputVectors.append(np.array(0))
      elif dir_split[-2] == 'scissors':
        outputVectors.append(np.array(2))

    data = {'input': inputImages, 'label': outputVectors}

    if self.transform:
      data = self.transform(data)

    return data


class ToTensor(object):
  def __call__(self, data):
    label, input = data['label'], data['input']
    size = 220
    input_tensor = torch.empty(len(input),size,size)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)

    data = {'label': label_tensor.long(), 'input': input_tensor}

    return data


class TestDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):  # fdir, pdir, sdir, transform=None):

    self.transform = transform
    self.dir_name = data_dir
    self.name_list = os.listdir(data_dir)

  def __len__(self):
    return len(self.name_list)

  def __getitem__(self, index):
    self.img_name = self.name_list[index]
    return self.img_name

  def custom_collate_fn_test(self, data):
    size = 220
    inputImages = []
    filename = []

    for sample in data:
      filename.append(sample)
      prs_img = cv2.imread(os.path.join(self.dir_name + sample))

      # Preprocessing
      hsvim = cv2.cvtColor(prs_img, cv2.COLOR_BGR2YCrCb)
      lower = np.array([0, 133, 77], dtype="uint8")
      upper = np.array([255, 173, 127], dtype="uint8")
      skinRegionHSV = cv2.inRange(hsvim, lower, upper)
      blurred = cv2.blur(skinRegionHSV, (2, 2))
      ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

      thresh = cv2.resize(thresh, (size, size), interpolation=cv2.INTER_CUBIC)

      gaussian = np.random.normal(0, 10, (size, size))
      thresh = thresh + gaussian
      cv2.normalize(thresh, thresh, 0, 255, cv2.NORM_MINMAX, dtype=-1)

      gray_img = thresh
      gray_img = gray_img[:, :, np.newaxis]

      inputImages.append(gray_img)

    data = {'input': inputImages, 'filename': filename}

    if self.transform:
      data = self.transform(data)
    return data


class ToTensor_test(object):
  def __call__(self, data):
    input, filename = data['input'], data['filename']
    size = 220
    input_tensor = torch.empty(len(input), size, size)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)

    data = {'input': input_tensor, 'filename': filename}

    return data