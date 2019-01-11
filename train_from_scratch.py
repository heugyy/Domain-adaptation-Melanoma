"""
Training a Classifier from scratch
=====================

1. Loading and normalizing dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import torch
from torchvision import datasets, models
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import time
import copy
from torch.optim import lr_scheduler
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark=True

batch_sz = 32
num_epoch = 50
init_learning_rate = 0.001
learning_rate_decay_factor = 0.1
num_epochs_decay = 30
dataset_name = 'Molemap_7cls'
train_dir = 'Data/comparason/transfer_learning/Train_Molemap'
test_dir = 'Data/comparason/transfer_learning/Test_Molemap'

# Data Directory
result_dir = 'checkpoints/' +dataset_name + '_FromScratch_Ep_'+str(num_epoch)+'_ILR_'+str(init_learning_rate) + '_DF_' + str(learning_rate_decay_factor) + '_NmD_' + str(num_epochs_decay)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

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

trainset = datasets.ImageFolder(train_dir, data_transforms['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
train_sizes = len(trainset)

testset = datasets.ImageFolder(test_dir, data_transforms['val'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
test_sizes = len(testset)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

classes = trainset.classes
class_num = len(classes)


########################################################################
# 2. Use resnet model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
net = models.resnet152(pretrained=False)
net = net.to(device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = init_learning_rate)
# Decay LR by a factor of learning_rate_decay_factor every num_epochs_decay epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs_decay, gamma=learning_rate_decay_factor)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
since = time.time()
file = open(result_dir + "/result.txt", "w")

file.write('==============================\n')
file.write('Parameters:\n')
file.write('batch_sz = : {:d}\n'.format(batch_sz))
file.write('num_epoch = : {:d}\n'.format(num_epoch))
file.write('init_learning_rate = : {:f}\n'.format(init_learning_rate))
file.write('learning_rate_decay_factor = : {:f}\n'.format(learning_rate_decay_factor))
file.write('num_epochs_decay = : {:d}\n'.format(num_epochs_decay))
file.write('==============================\n')

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0
best_acc_each_cls = list(0 for i in range(class_num))
loss_values = []

for epoch in range(num_epoch):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch, num_epoch - 1))

    scheduler.step()
    net.train()

    # initialize loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.set_grad_enabled(True):
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


    train_loss = running_loss / train_sizes
    train_acc = running_corrects.double() / train_sizes
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(
            train_loss, train_acc))
    file.write('Train Loss: {:.4f} Acc: {:.4f}\n'.format(train_loss, train_acc))


    net.eval()
    running_loss = 0.0
    running_corrects = 0

    class_correct = list(0 for i in range(class_num))
    class_total = list(0 for i in range(class_num))
    # Iterate over data.
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (preds[i].item() == label)
            class_total[label] += 1
    test_loss = running_loss / test_sizes
    test_acc = running_corrects.double() / test_sizes

    save_name = result_dir + '/' + str(epoch) + '_epoch.ckpt'
    torch.save(net.state_dict(), save_name)

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(
            test_loss, test_acc))
    file.write('Test Loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))

    val_acc_each_class = [class_correct[i] / class_total[i] for i in range(class_num)]
    for i in val_acc_each_class:
        print('{:.4f}'.format(i), end=' ')
        file.write('{:.4f}'.format(i) + ' ')
    print()
    file.write('\n')

    # deep copy the model
    if test_acc > best_acc:
        best_acc = test_acc
        best_acc_each_cls = val_acc_each_class
        best_model_wts = copy.deepcopy(net.state_dict())

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
# load best model weights
net.load_state_dict(best_model_wts)
result_dir += '/best_model.ckpt'
torch.save(net.state_dict(), result_dir)


