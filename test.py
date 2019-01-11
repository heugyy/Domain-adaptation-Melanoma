######################################################################
## Testing on a test dataset using a pre-trained model
## print out the total accuracy and accuracy for each class
##
##
######################################################################

import torch
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_sz = 32

#####################
# data path #
#####################
data_dir ='Data/comparason/Modality_Domain_Adapt/MoleMap_dermo_test'  #the folder of the test data
model_dir = 'Result/Modality_DA/camera_augmentation/FakeCamera_aug_10_4000_Ep_50_ILR_0.0001_DF_0.1_NmD_30/best_model.ckpt'

######################################################################
# Load Data
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

image_datasets = datasets.ImageFolder(data_dir,data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_sz,
                                             shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
class_num = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# load model
the_model =  models.resnet152()
num_ftrs = the_model.fc.in_features
the_model.fc = nn.Linear(num_ftrs,class_num)
#if there are two gpus installed use this to map the model from gpu1 to gpu0
the_model.load_state_dict(torch.load(model_dir,map_location={'cuda:1': 'cuda:0'})) 
the_model = the_model.to(device)


the_model.eval()  # Set model to evaluate mode

running_corrects = 0

class_correct = list(0 for i in range(class_num))
class_total = list(0 for i in range(class_num))
tmp = 0
# Iterate over data.
for inputs, labels in dataloaders:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    # track history if only in train
    torch.set_grad_enabled(False)
    outputs = the_model(inputs)
    _, preds = torch.max(outputs, 1)


    # statistics
    running_corrects += torch.sum(preds == labels.data)

    for i in range(len(labels)):
        label = labels[i].item()
        class_correct[label] += (preds[i].item() == label)
        class_total[label] += 1

epoch_acc = running_corrects.double() / dataset_sizes

print('Acc: {:.4f}'.format( epoch_acc))

val_acc_each_class = [class_correct[i] / class_total[i] for i in range(class_num)]
for i in val_acc_each_class:
    print('{:.4f}'.format(i), end=' ')
print()



