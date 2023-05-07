# %%
# Import Libraries
import numpy as np

# PyTorch
import torch
import torchvision.transforms as tvt
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss

# Data Processing
from PIL import Image
import os
import cv2
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# GLOBAL VARIABLES
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# %% [markdown]
# # Save Annotations into Dictionary

# %%
def ImageProcessing(images):
    # Get dir
    image_dir = images["filepath"]

    # Get inputs
    image = Image.open(image_dir)
    bbox = [images["x1"], images["y1"], images["width"], images["height"]]

    # Normalize image
    toTensor = tvt.ToTensor()(image)
    toNormalize = tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(toTensor)
    return toNormalize, bbox

# %%
# Create Dataset
class MyDataset(torch.utils.data.Dataset):
    # Obtain meta information (e.g. list of file names) 
    def __init__(self, imagesDataFrame, class_list):
        super().__init__()
        self.imagesDataFrame = imagesDataFrame
        self.catEncoding = { # Set Integer Values for Cat
            "bus" : 0,
            "cat" : 1,
            "pizza" : 2
        }

    def __len__(self):
        return len(self.imagesDataFrame)
    
    def __getitem__(self, i):
        imagesDataFrame = self.imagesDataFrame.iloc[i]
        normalImage, bbox = ImageProcessing(imagesDataFrame)
        label = self.catEncoding[imagesDataFrame["category"]]

        # Fix box of resize
        width, height = bbox[2], bbox[3]
        bboxAdjust = [bbox[0], bbox[1], width, height]
        bboxTensor = torch.tensor(bboxAdjust, dtype=torch.float)
        return normalImage, label, bboxTensor

# %% [markdown]
# # Skip Block

# %%
# Based on DLStudio SkipBlock Code
# Based off following https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class Block(nn.Module):
      def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
          super(Block, self).__init__()
          self.downsample = downsample
          self.in_ch = in_ch
          self.out_ch = out_ch

          self.skip_connections = skip_connections
          
          self.conv1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU())
          self.conv2 = nn.Sequential(
                            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
                            nn.BatchNorm2d(out_ch))
          self.relu = nn.ReLU()

          if downsample:
              self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

      def forward(self, x):
          residual = x                                     
          out = self.conv1(x)
          
          if self.in_ch == self.out_ch:
              out = self.conv1(out)                              

          if self.downsample:
              out = self.downsampler(out)
              residual = self.downsampler(residual)
              
          if self.skip_connections:
              if self.in_ch == self.out_ch:
                  out = out + residual                              
              else:
                  # Assuming equivalent dimensions which this dataset fits
                  firstSection = out[:,:self.in_ch,:,:]
                  secondSection = out[:,self.in_ch:,:,:]

                  out = torch.cat((firstSection + residual, secondSection + residual), dim=1)

          return out

# %% [markdown]
# # CNN

# %%
# CNN model # Inspired by DLStudio LOADnet2 && notes from class
class CNN(nn.Module):
    def __init__(self, skip_connections=True, depth=8):
        super(CNN, self).__init__()
        self.skip_connections = skip_connections
        self.depth = depth // 2

        # Create base layers
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)

        # Create Depth layers 64
        self.skip64_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip64_arr.append(Block(in_ch=64, out_ch=64, skip_connections=self.skip_connections))

        self.skip64ds = Block(in_ch=64, out_ch=64, downsample=True, skip_connections=self.skip_connections)
        self.skip64to128 = Block(in_ch=64, out_ch=128, skip_connections=self.skip_connections)
        
        # Create depth layer 128
        self.skip128_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip128_arr.append(Block(in_ch=128, out_ch=128, skip_connections=self.skip_connections))
        self.skip128ds = Block(in_ch=128, out_ch=128, downsample=True, skip_connections=self.skip_connections)

        # Linear Layers #Limited from original file do to lack of memory space
        self.fc1 = nn.Linear(in_features=32*32*128, out_features=3) # 3 categories

        ## For Regression
        self.conv_sequential = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features=64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True))
        
        self.fc_sequential = nn.Sequential(nn.Linear(in_features=128*128*64, out_features=4)) # 4 dimensions bbox
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))

        # Classification
        cls = x.clone()
        for idx, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            cls = skip64(cls)
        cls = self.skip64ds(cls)
        for idx, skip64 in enumerate(self.skip64_arr[self.depth//4:]):
            cls = skip64(cls)
        cls = self.bn1(cls)
        cls = self.skip64to128(cls)
        for idx, skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            cls = skip128(cls)
        cls = self.bn2(cls)
        cls = self.skip128ds(cls)
        for idx, skip128 in enumerate(self.skip128_arr[self.depth//4:]):
            cls = skip128(cls)
        cls = cls.view(-1, 32 * 32 * 128)
        cls = self.fc1(cls)

        # Regression for BBox
        bbox = self.conv_sequential(x)
        bbox = bbox.view(x.size(0), -1)
        bbox = self.fc_sequential(bbox)

        return cls, bbox

# %% [markdown]
# # Training

# %%
############################## TRAINING ##############################
# Inspired by DLStudio run_code_for_training_with_CrossEntropy_and_MSE_Losses && notes from class
def trainMSERegression(model, trainDataLoader, runn_avg_size, epochs, lr=1e-4, betas=(0.9, 0.99)):
    model = model.to(device)
    num_layers = len(list(model.parameters()))
    print(f"Number of layers: {0}".format(num_layers))

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    labeling_loss_running_avg = []
    regression_loss_running_avg = []

    for epoch in range(1, epochs+1):
        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for batch_idx, (imgTensor, labels, bbox) in enumerate(trainDataLoader):
            imgTensor = imgTensor.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device)

            optimizer.zero_grad()
            outputs = model(imgTensor)
            output_label = outputs[0]
            output_bbox = outputs[1]

            loss_label = classification_criterion(output_label, labels)
            running_loss_labeling += loss_label.item()
            loss_label.backward(retain_graph=True)

            loss_bbox = regression_criterion(output_bbox, bbox)
            running_loss_regression += loss_bbox.item()
            loss_bbox.backward()

            optimizer.step()

            if(batch_idx % runn_avg_size == (runn_avg_size - 1)):
                labeling_loss_running_avg.append(running_loss_labeling / float(runn_avg_size))
                regression_loss_running_avg.append(running_loss_regression / float(runn_avg_size))

                running_loss_labeling = 0.0
                running_loss_regression = 0.0

    return labeling_loss_running_avg, regression_loss_running_avg

# %%
############################## TRAINING ##############################
# Inspired by DLStudio run_code_for_training_with_CrossEntropy_and_MSE_Losses && notes from class
def trainCompleteBoxIOU(model, trainDataLoader, runn_avg_size, epochs, lr=1e-4, betas=(0.9, 0.99)):
    model = model.to(device)
    num_layers = len(list(model.parameters()))
    print(f"Number of layers: {0}".format(num_layers))

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = complete_box_iou_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    labeling_loss_running_avg = []
    regression_loss_running_avg = []

    for epoch in range(1, epochs+1):
        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for batch_idx, (imgTensor, labels, bbox) in enumerate(trainDataLoader):
            imgTensor = imgTensor.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device)

            optimizer.zero_grad()
            outputs = model(imgTensor)
            output_label = outputs[0]
            output_bbox = outputs[1]

            loss_label = classification_criterion(output_label, labels)
            loss_label.backward(retain_graph=True)
            running_loss_labeling += loss_label.item()

            loss_bbox = regression_criterion(output_bbox, bbox, reduction = "mean")
            loss_bbox.backward()
            running_loss_regression += loss_bbox.item()

            optimizer.step()

            if(batch_idx % runn_avg_size == (runn_avg_size - 1)):
                labeling_loss_running_avg.append(running_loss_labeling / float(runn_avg_size))
                regression_loss_running_avg.append(running_loss_regression / float(runn_avg_size))

                running_loss_labeling = 0.0
                running_loss_regression = 0.0

    return labeling_loss_running_avg, regression_loss_running_avg

# %% [markdown]
# # Testing

# %%
def test(model, testDataLoader, class_list):
    model = model.to(device)
    confusion_matrix = np.zeros((len(class_list), len(class_list)))
    image_data = []

    with torch.no_grad():
        for imgTensor, labels, bbox in testDataLoader:
            # Set to Device
            imgTensor = imgTensor.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device)

            # Set Outputs
            outputs = model(imgTensor)
            output_label = outputs[0]
            output_bbox = outputs[1].tolist()

            _, predicted = torch.max(output_label, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1

            for img, original_label, predicted_label, original_bbox, predicted_bbox in zip(imgTensor, labels, predicted, bbox, output_bbox):
                image_data.append([img, original_label, predicted_label, original_bbox, predicted_bbox])

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy, image_data

# %% [markdown]
# # Validation and Plotting

# %%
def plotRegressionLosses(regression, lossType):
    figure = plt.figure(1)

    plt.plot(range(len(regression)), regression, label="Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    plt.savefig("lossvsiter_regress{0}.jpg".format(lossType))

# %%
def plotLabelLosses(labeling, lossType):
    figure = plt.figure(2)

    plt.plot(range(len(labeling)), labeling, label="Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    plt.savefig("lossvsiter_class{0}.jpg".format(lossType))

# %%
def plotConfusionMatrix(conf, accuracy, class_list, lossType):
    figure = plt.figure(3)
    sns.heatmap(conf, xticklabels=class_list, yticklabels=class_list, annot=True)
    plt.xlabel("True Label: Accuracy {0}".format(accuracy))
    plt.ylabel("Predicted Label")
    plt.savefig("confMatrix{0}.jpg".format(lossType))

# %%
def plotImages(image_data, lossType):
    fig, ax = plt.subplots(3, 3)
    row, col = 0, 0
    
    for idx, arr in enumerate(image_data):
        image = tvt.ToPILImage()(arr[0])

        invert_categories = { # Set Integer Values for Cat
            0 : "bus",
            1 : "cat",
            2 : "pizza"
        }

        original_cat = invert_categories[arr[1].item()]
        predicted_cat = invert_categories[arr[2].item()]
        original_bbox = arr[3]
        predicted_bbox = arr[4]

        # Original bbox
        #print(original_bbox)
        [x, y, w, h] = original_bbox
        image = np.uint8(image)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(36, 255, 12), thickness=2)
        image = cv2.putText(image, original_cat, (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)

        [x, y, w, h] = predicted_bbox
        # image = np.uint8(image)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(255, 36, 12), thickness=2)
        image = cv2.putText(image, predicted_cat, (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 36, 12), thickness=2)
        
        #print(row, col)
        ax[row, col].imshow(image)

        # Increment through row
        col += 1
        if col == 3: 
            col = 0
            row += 1
        
        if idx > 7:
            break

    # fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10), dpi = 150)
    # plt.show()
    # axs = axs.flatten()
    plt.savefig("predvsorginal_bbox{0}.jpg".format(lossType))

# %% [markdown]
# # Main Run Code

# %%
# Input
train_json = 'annotations/instances_train2014.json'
val_json = 'annotations/instances_val2014.json'

# Images saved directory
train_data_path = 'trainingData'
val_data_path = 'valData'

class_list = ['pizza', 'bus', 'cat']

trainDataFrame = pd.read_csv('train_labels.csv')
valDataFrame = pd.read_csv('val_labels.csv')

# %%
trainDataset = MyDataset(trainDataFrame, class_list)
valDataset = MyDataset(valDataFrame, class_list)

trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=2, num_workers=2, shuffle=True)
valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=2, num_workers=2, shuffle=True)

# %%
# Save networks
net = CNN()

# Train Networks
epochs=7
labelMSELoss, regressMSELosses = trainMSERegression(net, trainDataloader,  100, epochs=epochs, lr=1e-4, betas=(0.9, 0.99))
lossType = 'MSELoss'
plotRegressionLosses(regressMSELosses, lossType)
plotLabelLosses(labelMSELoss, lossType)

conf, acc, image_data = test(net, valDataloader, class_list)
plotConfusionMatrix(conf, acc, class_list, lossType)
plotImages(image_data, lossType)

labelCBIOULoss, regressCBIOULosses = trainCompleteBoxIOU(net, trainDataloader,  100, epochs=epochs, lr=1e-4, betas=(0.9, 0.99))
lossType = "CIoULoss"
plotRegressionLosses(regressCBIOULosses, lossType)
plotLabelLosses(labelCBIOULoss, lossType)

conf, acc, image_data = test(net, valDataloader, class_list)
plotConfusionMatrix(conf, acc, class_list, lossType)
plotImages(image_data, lossType)