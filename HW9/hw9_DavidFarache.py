# Import Libraries
import numpy as np
import torch
import torchvision.transforms as tvt
import torch.utils.data 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
from pprint import pprint
import seaborn as sns
import cv2
from ViTHelper import MasterEncoder
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
import time
import datetime

device = "cuda"
device = torch.device("cuda")

# GLOBAL VARIABLES
categories = ["airplane", "bus", "cat", "dog", "pizza"]
num_classes = len(categories)
class_encoding = {0: "airplane", 1: "bus", 2: "cat", 3: "dog", 4: "pizza"}

patch_size = 16 # pixels
C_in = 3
embedded_size = 64
max_seq_length = patch_size + 1 # class token
image_size = 64 # h x w

images_Train = '/scratch/gilbreth/dfarache/ece60146/David/HW9/trainingData'
images_Val = '/scratch/gilbreth/dfarache/ece60146/David/HW9/valData'

# DataLoader

def ImageProcessing(directory, cat):
    dir = os.path.join(directory, cat)

    imageFile = [image for image in os.listdir(dir)]
    imagesPIL = [Image.open(os.path.join(dir, image)).convert("RGB") for image in imageFile]

    toTensor = [tvt.ToTensor()(image) for image in imagesPIL]
    toNormalize = [tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor) for tensor in toTensor]
    return toNormalize

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, categories):
        super().__init__()
        self.categories = categories
        self.root = root
        self.data = []
        
        for i, cat in enumerate(self.categories):
            images = ImageProcessing(self.root, cat)
            for image in images:
                self.data.append([image, i])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        image = self.data[i][0]
        label = torch.tensor(self.data[i][1])
        return image, label

# Network

# Based on https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
# Based on https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c 

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embedded_size, image_size, in_channels=3):
        super(PatchEmbedding, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedded_size = embedded_size
        self.image_size = image_size
        
        self.sequential = nn.Sequential(
            # batch_size, embedded_size, patch_height, patch_width
            nn.Conv2d(in_channels, embedded_size, kernel_size=patch_size, stride=patch_size), 
            
            # prior to batch_size, patch_height * patch_width, embedded_size
            Rearrange('b e (h) (w) -> b (h w) e') 
        )
        
        self.class_token = nn.Parameter(torch.rand(1, 1, self.embedded_size))
        self.positions = nn.Parameter(torch.randn((self.image_size // self.patch_size)**2 + 1, self.embedded_size)) 

    def forward(self, x):
        x = self.sequential(x)
        # Repeat batch
        class_tokens = repeat(self.class_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([class_tokens, x], dim=1) # prepend token to match sequence len, give location
        x = x + self.positions
        return x

# Based on https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class ClassificationHead(nn.Sequential):
    def __init__(self, embedded_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.Linear(embedded_size, num_classes))

# Assignment Network

# Based on https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class ViT_Assignment(nn.Sequential):
    def __init__(self, how_many_basic_encoders=2, num_attention_heads=2, in_channels=3):
        super(ViT_Assignment, self).__init__(
            PatchEmbedding(patch_size, embedded_size, image_size, in_channels),
            MasterEncoder(max_seq_length, embedded_size, how_many_basic_encoders, num_attention_heads),
            ClassificationHead(embedded_size, num_classes) 
        )

# Extra Credit Network

# Based on https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class ViT_ExtraCredit(nn.Sequential):
    def __init__(self, how_many_basic_encoders=2, num_attention_heads=2, in_channels=3):
        super(ViT_ExtraCredit, self).__init__(
            PatchEmbedding(patch_size, embedded_size, image_size, in_channels), 
            MasterEncoder(max_seq_length, embedded_size, how_many_basic_encoders, num_attention_heads, method=2),
            ClassificationHead(embedded_size, num_classes)
        )

# Training

def train(net, epochs, lr, dataloader):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    lossRun = []

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if((i + 1) % 100 == 0):
                print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch, i + 1, running_loss / 100))
                lossRun.append(running_loss / 100)
                running_loss = 0.0
    
    return lossRun

# Testing

def test(net, dataloader, numCat):
    net = net.to(device)
    confusion_matrix = np.zeros((numCat, numCat))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1
            
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy

# Plotting

def trainingPlot(loss, epochs):
    plt.plot(range(len(loss)), loss)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.show()

def confusionMatrix(conf, categories, acc):
    sns.heatmap(conf, xticklabels=categories, yticklabels=categories, annot=True)
    plt.xlabel(f"True Label \n Accuracy: {acc}")
    plt.ylabel("Predict Label")
    plt.show()

# Main

## DataLoaders

batch_size = 16

# Create DataLoaders
trainDataset = MyDataset(images_Train, categories)
trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

testDataset = MyDataset(images_Val, categories)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

## Parameters for Training

lr = 1e-4 # Learning Rate
epochs = 20 # Epochs

## Assignment

net1 = ViT_Assignment()
training_loss = train(net1, epochs, lr, trainDataloader)
trainingPlot(training_loss, epochs)

confusion_matrix, accuracy = test(net1, testDataloader, num_classes)
confusionMatrix(confusion_matrix, categories, accuracy)

## Extra Credit

net2 = ViT_ExtraCredit()
training_loss = train(net2, epochs, lr, trainDataloader)
trainingPlot(training_loss, epochs)

confusion_matrix, accuracy = test(net2, testDataloader, num_classes)
confusionMatrix(confusion_matrix, categories, accuracy)