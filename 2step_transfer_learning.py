#######################################################################
##
## 2-step transfer learning
## save model after each epoch and the best model
## save all the results in a txt file
##
#######################################################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark=True

#####################
# Training Flags #
#####################
batch_sz = 32
num_epoch = 50
init_learning_rate = 0.0001
learning_rate_decay_factor = 0.2
num_epochs_decay = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epoch):
    since = time.time()
    file = open(result_dir + "/result.txt", "w")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_each_cls = [0.0] * class_num
    loss_values =[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            class_correct = list(0 for i in range(class_num))
            class_total = list(0 for i in range(class_num))
            tmp =0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (preds[i].item() == label)
                    class_total[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            save_name = result_dir+'/'+str(epoch) + '_epoch.ckpt'
            torch.save(model.state_dict(), save_name)

            loss_values.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            file.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            val_acc_each_class = [class_correct[i] / class_total[i] for i in range(class_num)]
            for i in val_acc_each_class:
                print('{:.4f}'.format(i), end=' ')
                file.write('{:.4f}'.format(i)+' ')

            print()
            file.write('\n')
            if phase == 'val':
                # deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_acc_each_cls = val_acc_each_class
                    best_model_wts = copy.deepcopy(model.state_dict())

    # print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(*best_acc_each_cls)
    file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    file.write('Best val Acc: {:4f}\n'.format(best_acc))
    for i in best_acc_each_cls:
        file.write('{:.4f}'.format(i) + ' ')
    file.write('\n')
    file.close()
    file_best_result.write(fd_name+': ' +str(best_acc) + ' ')
    for i in best_acc_each_cls:
        file_best_result.write('{:.4f}'.format(i) + ' ')
    file_best_result.write('{:.0f}m{:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # return model,best_acc, best_acc_each_cls
    return model

######################################################################
# 1st Step Transfer learning 
######################################################################

#################
# data path
Data_dir0 = 'Data/comparason/transfer_learning/' #Root directory
train_dir = Data_dir0 + '/Train_Molemap'
test_dir = Data_dir0 + '/Test_Molemap'
fd_name = 'ImgNt_MlMp_ISIC1'
file_best_result = open(Data_dir0 + "/result.txt", "w")

result_dir = 'checkpoints/transfer_learning/'+ fd_name+'_Ep_' + str(num_epoch) + '_ILR_' + str(
    init_learning_rate) + '_DF_' + str(learning_rate_decay_factor) + '_NmD_' + str(num_epochs_decay)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
##################
# Load Data
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(test_dir, data_transforms['val'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sz,
    shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
class_num = len(class_names)

###################
# Training and Testing
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, class_num)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
pm = model_ft.parameters()

optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=init_learning_rate)
# Decay LR by a factor of learning_rate_decay_factor every num_epochs_decay epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs_decay,
                                       gamma=learning_rate_decay_factor)

#################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epoch)
result_dir_s = result_dir + '/best_model.ckpt'
torch.save(model_ft.state_dict(), result_dir_s)

######################################################################
# 2nd Step Transfer learning 
######################################################################

#################
# data path
train_dir = Data_dir0 + '/ISIC_train'
test_dir = Data_dir0 + '/ISIC_test'
fd_name = 'ImgNt_MlMp_ISIC2'
result_dir = 'checkpoints/transfer_learning/'+ fd_name+'_Ep_' + str(num_epoch) + '_ILR_' + str(
    init_learning_rate) + '_DF_' + str(learning_rate_decay_factor) + '_NmD_' + str(num_epochs_decay)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
##################
# Load Data
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Data augmentation and normalization for training
# Just normalization for validation
image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(test_dir, data_transforms['val'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sz,
    shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
class_num = len(class_names)

###################
# Training and Testing
model_ft.fc = nn.Linear(num_ftrs, class_num)
model_ft = model_ft.to(device)
pm = model_ft.parameters()

optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=init_learning_rate)
# Decay LR by a factor of learning_rate_decay_factor every num_epochs_decay epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs_decay,
                                       gamma=learning_rate_decay_factor)

#################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epoch)
result_dir_s = result_dir + '/best_model.ckpt'
torch.save(model_ft.state_dict(), result_dir_s)
file_best_result.close()
