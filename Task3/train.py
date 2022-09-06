import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2

from dataset import *
from copy import copy

import warnings

warnings.filterwarnings('ignore')


def train(args):
  dataset_train = CustomDataset(args.train_dir, transform=transforms.Compose([ToTensor()]))
  loader_train = DataLoader(dataset_train, batch_size=args.batchsize, \
                            shuffle=True, collate_fn=dataset_train.custom_collate_fn_train, num_workers=8)

  dataset_val = CustomDataset(args.val_dir, transform=transforms.Compose([ToTensor()]))
  loader_val = DataLoader(dataset_val, batch_size=args.batchsize, \
                          shuffle=True, collate_fn=dataset_val.custom_collate_fn_validation, num_workers=8)

  num_train = len(dataset_train)*20
  num_val = len(dataset_val)

  # Define Model
  model = mobilenet_v2()
  model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
  model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=3)

  soft = nn.Softmax(dim=1)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Current device:", device)
  
  model.to(device)

  # Define the loss, optimizer, scheduler
  criterion = nn.CrossEntropyLoss().to(device)
  optim = torch.optim.Adam(model.parameters(), lr = 0.001)

  best_epoch = 0
  accuracy_save = np.array(0)
  for epoch in range(args.epochs):

    model.train()
    train_loss = []
    correct_train = 0
    correct_val = 0
    correct_batch = 0
    val_loss = 0

    for batch, data in enumerate(loader_train, 1):
      label = data['label'].to(device)
      input = data['input'].to(device)

      output = model(input)
      label_pred = soft(output).argmax(1)
  
      optim.zero_grad()
  
      loss = criterion(output, label)
      loss.backward()
  
      optim.step()

      correct_train += (label == label_pred).float().sum()
  
      train_loss += [loss.item()]

    accuracy_train = correct_train / num_train
  
    correct_val = 0
    accuracy_tmp = np.array(0)
    # with torch.no_grad():
    #
    #   model.eval()
    #   val_loss = []
    #   for batch, data in enumerate(loader_val, 1):
    #
    #     label_val = data['label'].to(device)
    #     input_val = data['input'].to(device)
    #
    #     output_val = model(input_val)
    #
    #     label_val_pred = soft(output_val).argmax(1)
    #
    #     correct_val += (label_val == label_val_pred).float().sum()
    #
    #     loss = criterion(output_val, label_val)
    #     val_loss += [loss.item()]
    #
    #   accuracy_val = correct_val / num_val
    #
    #   # Save the best model wrt val accuracy
    #   accuracy_tmp = accuracy_val.cpu().numpy()
    #   if accuracy_save < accuracy_tmp:
    #     best_epoch = epoch
    #     accuracy_save = accuracy_tmp.copy()
    #     torch.save(model.state_dict(), 'param.data')
    #     print(".......model updated (epoch = ", epoch+1, ")")
    accuracy_val = 0
    print("epoch: %04d / %04d | train loss: %.5f | train accuracy: %.4f | validation loss: %.5f | validation accuracy: %.4f" %
          (epoch+1, args.epochs, np.mean(train_loss), accuracy_train, np.mean(val_loss), accuracy_val))

  print("Model with the best validation accuracy is saved.")
  print("Best epoch: ", best_epoch)
  print("Best validation accuracy: ", accuracy_save)
  print("Done.")
