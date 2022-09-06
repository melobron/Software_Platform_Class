import os

import numpy as np
import torch
import torch.nn as nn

import natsort

from skimage.color import rgb2gray
import imageio

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

    lst_rock = [f for f in lst_rock if f.endswith(".png")]
    lst_paper = [f for f in lst_paper if f.endswith(".png")]
    lst_scissors = [f for f in lst_scissors if f.endswith(".png")]

    self.lst_dir = [self.rock_dir] * len(lst_rock) + [self.paper_dir] * len(lst_paper) + [self.scissors_dir] * len(lst_scissors)
    self.lst_prs = natsort.natsorted(lst_rock) + natsort.natsorted(lst_paper) + natsort.natsorted(lst_scissors)

  def __len__(self):
    return len(self.lst_prs)

  def __getitem__(self, index): 
    self.img_dir = self.lst_dir[index]
    self.img_name = self.lst_prs[index]

    return [self.img_dir, self.img_name] 
    
  def custom_collate_fn(self, data):

    inputImages = []
    outputVectors = []

    for sample in data:
      prs_img = imageio.imread(os.path.join(sample[0] + sample[1]))
      gray_img = rgb2gray(prs_img)

      if gray_img.ndim == 2:
        gray_img = gray_img[:, :, np.newaxis]

      inputImages.append(gray_img.reshape(300,300,1))

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.append(np.array(0))
      elif dir_split[-2] == 'paper':
        outputVectors.append(np.array(1))
      elif dir_split[-2] == 'scissors':
        outputVectors.append(np.array(2))

    data = {'input': inputImages, 'label': outputVectors}

    if self.transform:
      data = self.transform(data)

    return data


class ToTensor(object):
  def __call__(self, data):
    label, input = data['label'], data['input']

    input_tensor = torch.empty(len(input),300,300)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)
    input_tensor = torch.nn.functional.interpolate(input_tensor, (89, 100), mode='bicubic')
    data = {'label': label_tensor.long(), 'input': input_tensor}

    return data

