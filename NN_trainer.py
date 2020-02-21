from collections import defaultdict
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import matplotlib.pyplot as plt
import time
import copy
from torchvision import transforms,models
import torch.nn as nn
from os import walk



class Train_model():
    def __init__(self,train_dir,
                 val_dir,
                 model,
                 batch_sizes=list(),
                 epoch_numbers=list(),
                optimizer=list()):
        self.model=model
        self.batch_sizes=list(batch_sizes)
        self.epoch_numbers=list(epoch_numbers)
        self.optimizer=list(optimizer)
        self.train_dir=train_dir
        self.val_dir=val_dir

        self.trainer()

    def transform_loading(self):

        train_transforms=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        val_transforms=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


        train_dataset = torchvision.datasets.ImageFolder(self.train_dir, train_transforms)
        val_dataset = torchvision.datasets.ImageFolder(self.val_dir, val_transforms)

        return train_dataset, val_dataset

    def model_parameters(self,model,number_of_classes=2):

        self.number_of_classes=number_of_classes

        if model=='alexnet':
            model = models.alexnet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            number_of_filters = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(number_of_filters, self.number_of_classes)

        elif model=='vgg19':
            model = models.vgg19(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            number_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]  # Remove last layer
            features.extend([torch.nn.Linear(number_features, len(self.number_of_classes))])
            model.classifier = torch.nn.Sequential(*features)

        elif model=='resnet34':
            model = models.resnet34(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            number_of_filters = model.fc.in_features
            model.fc = nn.Linear(number_of_filters, self.number_of_classes)

        elif model =='vgg16':
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            number_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]  # Remove last layer
            features.extend([torch.nn.Linear(number_features, len(self.number_of_classes))])
            model.classifier = torch.nn.Sequential(*features)

        return model

    def trainer(self):

        loss_val = defaultdict(list)
        accuracy_val = defaultdict(list)




        for batch_size in self.batch_sizes: # batch size should be divided into number of images
            print(batch_size)
            print('*' * 30, f'This is batch size number {batch_size}')


            train_dataloader = torch.utils.data.DataLoader(
                self.transform_loading()[0], batch_size=batch_size, shuffle=True, num_workers=0)
            val_dataloader = torch.utils.data.DataLoader(
               self.transform_loading()[1], batch_size=batch_size, shuffle=False, num_workers=0)


            print('THIS IS MODEL NAME', self.model)
            # Display grad for all conv layers

            model=self.model_parameters(self.model)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            loss = torch.nn.CrossEntropyLoss()  # get neurons not probability
            for optimizer in self.optimizer:
                if optimizer == 'Adam':
                    print('*' * 20, 'This is Adam optimizer')
                    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
                    # Decay LR by a factor of 0.1 every 7 epochs
                    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

                    for epoch_number in self.epoch_numbers:
                        print('*' * 20, f'This is number of {self.epoch_numbers} epoch')
                        for epoch in range(epoch_number):
                            print('Epoch {}/{}:'.format(epoch, epoch_number - 1), flush=True)
                            # Each epoch has a training and validation phase
                            for phase in ['train', 'val']:
                                if phase == 'train':
                                    dataloader = train_dataloader
                                    #scheduler.step()
                                    model.train()  # Set model to training mode
                                else:
                                    dataloader = val_dataloader
                                    model.eval()  # Set model to evaluate mode

                                running_loss = 0.
                                running_acc = 0.

                                # Iterate over data.
                                for inputs, labels in tqdm(dataloader):
                                    inputs = inputs.to(device)
                                    labels = labels.to(device)

                                    optimizer.zero_grad()

                                    # forward and backward
                                    with torch.set_grad_enabled(phase == 'train'):
                                        preds = model(inputs)
                                        loss_value = loss(preds, labels)
                                        preds_class = preds.argmax(dim=1)

                                        # backward + optimize only if in training phase
                                        if phase == 'train':
                                            loss_value.backward()
                                            optimizer.step()

                                    # statistics
                                    running_loss += loss_value.item()
                                    running_acc += (preds_class == labels.data).float().mean()

                                epoch_loss = running_loss / len(dataloader)
                                epoch_acc = running_acc / len(dataloader)

                                # for each model you need to save everything in one place
                                if phase == 'val':
                                    accuracy_val[batch_size, epoch_number, 'Adam'].append(epoch_acc)
                                    loss_val[batch_size, epoch_number, 'Adam'].append(epoch_loss)

                                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)
                if optimizer == 'SGD':
                    print('*' * 20, "This is SGD optimizer")
                    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

                    # Decay LR by a factor of 0.1 every 7 epochs
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

                    for epoch_number in self.epoch_numbers:
                        print('*' * 20, f'This is number of {epoch_number}')
                        for epoch in range(epoch_number):
                            print('Epoch {}/{}:'.format(epoch, epoch_number - 1), flush=True)
                            # Each epoch has a training and validation phase
                            for phase in ['train', 'val']:
                                if phase == 'train':
                                    dataloader = train_dataloader
                                    #scheduler.step()
                                    model.train()  # Set model to training mode
                                else:
                                    dataloader = val_dataloader
                                    model.eval()  # Set model to evaluate mode

                                running_loss = 0.
                                running_acc = 0.

                                # Iterate over data.
                                for inputs, labels in tqdm(dataloader):
                                    inputs = inputs.to(device)
                                    labels = labels.to(device)

                                    optimizer.zero_grad()

                                    # forward and backward
                                    with torch.set_grad_enabled(phase == 'train'):
                                        preds = model(inputs)
                                        loss_value = loss(preds, labels)
                                        preds_class = preds.argmax(dim=1)

                                        # backward + optimize only if in training phase
                                        if phase == 'train':
                                            loss_value.backward()
                                            optimizer.step()

                                    # statistics
                                    running_loss += loss_value.item()
                                    running_acc += (preds_class == labels.data).float().mean()

                                epoch_loss = running_loss / len(dataloader)
                                epoch_acc = running_acc / len(dataloader)

                                # for each model you need to save everything in one place
                                if phase == 'val':
                                    accuracy_val[batch_size, epoch_number, 'SGD'].append(epoch_acc.item())
                                    loss_val[batch_size, epoch_number, 'SGD'].append(epoch_loss)

                                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)





        print('This is result of loss_val ', loss_val)
        print('This is result of accuracy_val ',accuracy_val)

        return loss_val, accuracy_val



traine_neural_network=Train_model(model='alexnet',
                  batch_sizes=[16,32],
                  epoch_numbers=[4,8],
                  optimizer=['Adam','SGD'],
                  train_dir='/Users/dariavolkova/Desktop/lab_future/1_crackdetection/craks_dataset/_0_dataset/train',
                  val_dir='/Users/dariavolkova/Desktop/lab_future/1_crackdetection/craks_dataset/_0_dataset/val')

traine_neural_network.trainer()



