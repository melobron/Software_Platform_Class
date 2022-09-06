import sys
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
from dataset import *


def main(test_dir, result):
  num_test = len(os.listdir(test_dir)) 

  ''' YOU SHOULD SUBMIT param.data FOR THE INFERENCE WITH TEST DATA'''
  model_ckpt = './param.data'

  transform = transforms.Compose([ToTensor_test()])
  dataset_test = TestDataset(test_dir, transform=transform)
  loader_test = DataLoader(dataset_test, batch_size=num_test, \
          collate_fn=dataset_test.custom_collate_fn_test, num_workers=8)

  # Define Model
  model = mobilenet_v2()
  model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
  model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=3)

  soft = nn.Softmax(dim=1)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Current device:", device)

  if model_ckpt:
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

  correct_test = 0
  model.eval()
  with torch.no_grad():
    model = model.cuda()
    test_loss = []

    for batch, data in enumerate(loader_test, 1):

      input_test = data['input'].to(device)
      fname_test = data['filename']
      output_test = model(input_test)
  
      label_test_pred = soft(output_test).argmax(1)
 
  label_test_pred = label_test_pred.cpu().numpy().tolist()

  f = open(result, "w")
  for i in range(len(fname_test)):
    f.write(fname_test[i] + ",")
    f.write(str(label_test_pred[i]) + "\n")
  f.close()
  print("Done.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
