import torch
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import ImageFile
from sklearn.metrics import roc_auc_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_sz = 32

#####################
# data path #
#####################
# data_dir = 'Data/sub/val'
# data_dir ='Data/MoleMap_test'
data_dir ='Data/comparason/Modality_Domain_Adapt/MoleMap_dermo_test'
# data_dir ='Data/comparason/Dataset_Domain_Adapt/MoleMap_dermo_test'
# data_dir = 'Data/comparason/transfer_learning/Test_Molemap'
# data_dir = 'Data/comparason/transfer_learning/ISIC_test'
# data_dir = 'MoleMap/classes/val'
# Model_dir = 'Result/Modality_DA/Molemap_25cls_Ep_50_ILR_0.001_DF_0.1_NmD_30/best_model.ckpt'
# Model_dir = 'Result/Modality_DA/ImgNt_MlMp_Camera_Ep_50_ILR_0.0001_DF_0.2_NmD_15/best_model.ckpt'
# Model_dir = 'checkpoints/Imgnet_MlMp_dermo_Ep_50_ILR_0.0001_DF_0.1_NmD_30/best_model.ckpt'
# Model_dir = 'checkpoints/Modality_DA/ImgNt_MlMp_Dermo_Ep_50_ILR_0.0001_DF_0.2_NmD_15/best_model.ckpt'
# Model_dir = 'Result/Molemap_25cls_Ep_50_ILR_0.001_DF_0.1_NmD_30/best_model.ckpt'
# Model_dir = 'checkpoints/MpCamera_aug_10_original_Ep_50_ILR_0.0001_DF_0.1_NmD_30/best_model.ckpt'
# Model_dir = 'Result/Dataset_DA/0MlMp_all_modality_augmentation/10_prc_1000fakeISIC_Ep_50_ILR_0.0001_DF_0.2_NmD_15/best_model.ckpt'
Model_dir = 'Result/Modality_DA/camera_augmentation/FakeCamera_aug_10_4000_Ep_50_ILR_0.0001_DF_0.1_NmD_30/best_model.ckpt'
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

image_datasets = datasets.ImageFolder(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_sz,
                                          shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
class_num = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
#
the_model = models.resnet152()
num_ftrs = the_model.fc.in_features
the_model.fc = nn.Linear(num_ftrs, class_num)
the_model.load_state_dict(torch.load(Model_dir,map_location={'cuda:1': 'cuda:0'}))
the_model = the_model.to(device)

the_model.eval()  # Set model to evaluate mode

MEL_corrects_mel_benign = 0
Negative_corrects_mel_benign = 0
MEL_condition_mel_benign = 0
Negative_condition_mel_benign = 0

MEL_corrects_cancer_non = 0
Negative_corrects_cancer_non = 0
MEL_condition_cancer_non = 0
Negative_condition_cancer_non = 0

prob_mel = np.array([], dtype=np.float32)
prob_cancer = np.array([], dtype=np.float32)
labels_mel_benign_total = np.array([], dtype=np.int64)
labels_cancer_non_total = np.array([], dtype=np.int64)

tmp = 0
# Iterate over data.
for inputs, labels in dataloaders:
    if class_num == 7:
        labels_mel_benign = torch.where(labels != 4, torch.tensor([0]), labels)
        labels_mel_benign = torch.where(labels_mel_benign == 4, torch.tensor([1]), labels_mel_benign)

        labels_cancer_non = torch.where(labels == 1, torch.tensor([1]), labels)
        labels_cancer_non = torch.where(labels_cancer_non == 4, torch.tensor([1]), labels_cancer_non)
        labels_cancer_non = torch.where(labels_cancer_non != 1, torch.tensor([0]), labels_cancer_non)

    if class_num == 25:  # Mel:17 BCC,SCC,19,20
        labels_mel_benign = torch.where(labels != 9, torch.tensor([0]), labels)
        labels_mel_benign = torch.where(labels_mel_benign == 9, torch.tensor([1]), labels_mel_benign)

        labels_cancer_non = torch.where(labels == 1, torch.tensor([0]), labels)
        labels_cancer_non = torch.where(labels_cancer_non == 9, torch.tensor([1]), labels_cancer_non)
        labels_cancer_non = torch.where(labels_cancer_non == 11, torch.tensor([1]), labels_cancer_non)
        labels_cancer_non = torch.where(labels_cancer_non == 13, torch.tensor([1]), labels_cancer_non)
        labels_cancer_non = torch.where(labels_cancer_non != 1, torch.tensor([0]), labels_cancer_non)
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels_mel_benign_total = np.concatenate((labels_mel_benign_total,np.array(labels_mel_benign)),axis=0)
    labels_cancer_non_total = np.concatenate((labels_cancer_non_total,np.array(labels_cancer_non)),axis=0)

    # forward
    # track history if only in train
    torch.set_grad_enabled(False)
    outputs = the_model(inputs)
    # SoftMax for outputs to transfer to probability space
    m = nn.Softmax(dim=1)
    outputs = m(outputs)

    if class_num == 7:
        prob_mel = np.concatenate((prob_mel,np.array(outputs[:, 4].clone())),axis=0)
        prob_cancer = np.concatenate((prob_cancer,np.array(outputs[:, 1].clone() + outputs[:, 4].clone())),axis=0)

    if class_num == 25:
        prob_mel = np.concatenate((prob_mel,np.array(outputs[:, 9].clone())),axis=0)
        prob_cancer = np.concatenate((prob_cancer,np.array(outputs[:, 9].clone() + outputs[:, 11].clone() + outputs[:, 13].clone())),axis=0)

    _, preds = torch.max(outputs, 1)
    preds = preds.cpu()
    if class_num == 7:
        preds_mel_benign = torch.where(preds != 4, torch.tensor([0]), preds)
        preds_mel_benign = torch.where(preds_mel_benign == 4, torch.tensor([1]), preds_mel_benign)

        preds_cancer_non = torch.where(preds == 1, torch.tensor([1]), preds)
        preds_cancer_non = torch.where(preds_cancer_non == 4, torch.tensor([1]), preds_cancer_non)
        preds_cancer_non = torch.where(preds_cancer_non != 1, torch.tensor([0]), preds_cancer_non)

    if class_num == 25:
        preds_mel_benign = torch.where(preds != 9, torch.tensor([0]), preds)
        preds_mel_benign = torch.where(preds_mel_benign == 9, torch.tensor([1]), preds_mel_benign)

        preds_cancer_non = torch.where(preds == 1, torch.tensor([0]), preds)
        preds_cancer_non = torch.where(preds_cancer_non == 9, torch.tensor([1]), preds_cancer_non)
        preds_cancer_non = torch.where(preds_cancer_non == 11, torch.tensor([1]), preds_cancer_non)
        preds_cancer_non = torch.where(preds_cancer_non == 13, torch.tensor([1]), preds_cancer_non)
        preds_cancer_non = torch.where(preds_cancer_non != 1, torch.tensor([0]), preds_cancer_non)
    preds_mel_benign = preds_mel_benign.to(device)
    preds_cancer_non = preds_cancer_non.to(device)

    for i in range(len(labels_mel_benign)):
        label = labels_mel_benign[i].item()
        if label == 1:
            MEL_corrects_mel_benign += (preds_mel_benign[i].item() == label)
        else:
            Negative_corrects_mel_benign += (preds_mel_benign[i].item() == label)

        label = labels_cancer_non[i].item()
        if label == 1:
            MEL_corrects_cancer_non += (preds_cancer_non[i].item() == label)
        else:
            Negative_corrects_cancer_non += (preds_cancer_non[i].item() == label)
    # statistics
    MEL_condition_mel_benign += torch.sum(labels_mel_benign.data == 1)
    Negative_condition_mel_benign += torch.sum(labels_mel_benign.data == 0)
    MEL_condition_cancer_non += torch.sum(labels_cancer_non.data == 1)
    Negative_condition_cancer_non += torch.sum(labels_cancer_non.data == 0)

SE_mel_benign = MEL_corrects_mel_benign / MEL_condition_mel_benign.double()
SP_mel_benign = Negative_corrects_mel_benign / Negative_condition_mel_benign.double()

auc_mel_non = roc_auc_score(labels_mel_benign_total, prob_mel)
auc_cancer_non = roc_auc_score(labels_cancer_non_total, prob_cancer)

print('MEL_corrects_mel_benign: {:.0f}, MEL_condition_mel_benign: {:.0f}'.format(MEL_corrects_mel_benign,
                                                                                 MEL_condition_mel_benign.double()))
print(
    'Negative_corrects_mel_benign: {:.0f}, MNegative_condition_mel_benign: {:.0f}'.format(Negative_corrects_mel_benign,
                                                                                          Negative_condition_mel_benign.double()))
print('Melanoma-vs-Benign Sensitivity: {:.4f} Specificity: {:.4f}'.format(SE_mel_benign, SP_mel_benign))
print('Melanoma-vs-Benign ROC-AUC: {:.4f}'.format(auc_mel_non))

SE_cancer_non = MEL_corrects_cancer_non / MEL_condition_cancer_non.double()
SP_cancer_non = Negative_corrects_cancer_non / Negative_condition_cancer_non.double()
print('MEL_corrects_cancer_non: {:.0f}, MEL_condition_cancer_non: {:.0f}'.format(MEL_corrects_cancer_non,
                                                                                 MEL_condition_cancer_non.double()))
print(
    'Negative_corrects_cancer_non: {:.0f}, Negative_condition_cancer_non: {:.0f}'.format(Negative_corrects_cancer_non,
                                                                                          Negative_condition_cancer_non.double()))
print('Cancer-vs-Nocancer Sensitivity: {:.4f} Specificity: {:.4f}'.format(SE_cancer_non, SP_cancer_non))
print('Cancer-vs-Nocancer ROC-AUC: {:.4f}'.format(auc_cancer_non))
print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(SE_mel_benign.double(), SP_mel_benign.double(),auc_mel_non, SE_cancer_non.double(),
                                           SP_cancer_non.double(),auc_cancer_non))

# val_acc_each_class = [class_correct[i] / class_total[i] for i in range(class_num)]
# for i in val_acc_each_class:
#     print('{:.4f}'.format(i), end=' ')
# print()



