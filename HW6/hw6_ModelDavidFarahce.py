# %%
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
import seaborn as sns
from torchvision.ops import box_convert
import pandas as pd
import cv2
import json
from pprint import pprint
from torchinfo import summary
from copy import deepcopy


def get_bbox_from_tensor(cell_row_index, cell_col_index, predicted_regression_vector):
    del_x,del_y = predicted_regression_vector[0], predicted_regression_vector[1]
    h,w = predicted_regression_vector[2], predicted_regression_vector[3]
    h *= yolo_interval
    w *= yolo_interval       
    
    bbox_center_x_coord = cell_col_index * yolo_interval  +  yolo_interval/2  +  del_x * yolo_interval
    bbox_center_y_coord = cell_row_index * yolo_interval  +  yolo_interval/2  +  del_y * yolo_interval
    x1 =  int(bbox_center_x_coord - w / 2.0)
    y1 =  int(bbox_center_y_coord - h / 2.0)
    
    return [x1, y1, int(w + x1), int(h + y1)]

# Set Initial Values

# GLOBAL VARIABLES
device = 'cuda'
device = torch.device(device)

# For DataLoader
root = '/scratch/gilbreth/dfarache/ece60146/David/HW6/'
class_list = ["bus", "cat", "pizza"]
inverse_categories = {6: "bus", 17: "cat", 59: "pizza"}

# Global YOLO Values (had to seperate due to no longer sharing training and yolo vecotr in same function)
num_anchor_boxes = 5 # (height/width)   1/5  1/3  1/1  3/1  5/1  aspect ratios
max_num_objects = len(class_list)

yolo_vector_size = 8 # []
yolo_interval = 42 # Each cell is 42x42 pixels

num_cells_image_width = 256 // yolo_interval
num_cells_image_height = 256 // yolo_interval

num_yolo_cells = num_cells_image_width * num_cells_image_height # Create a grid of cells overlaying the image

# YOLO Tensor

# Based on DLStudio run_code_for_training_multi_instance_detection function and homework example
def createYoloTensor(bboxs, labels, num_images_in_batch=1):
    yolo_tensor = torch.zeros(num_yolo_cells, num_anchor_boxes, yolo_vector_size)

    # Create empty torch tensors
    height_center_bb = torch.zeros(num_images_in_batch, 1).float()
    width_center_bb = torch.zeros(num_images_in_batch, 1).float()
    object_bb_height = torch.zeros(num_images_in_batch, 1).float()
    object_bb_width = torch.zeros(num_images_in_batch, 1).float()
    
    numericDict = {6: 0, 17: 1, 59: 2} # Swap id to index for one-hot encoding
   
    # i is index of object in the foreground
    for i in range(max_num_objects):   
        # remove .float     
        y_cord_center = (bboxs[i, 1] + bboxs[i, 3] / 2.0).int() # Get y-coordinate object center from y1
        x_cord_center = (bboxs[i, 0] + bboxs[i, 2] / 2.0).int() # Get x-coordinate object center from x1

        object_bb_height = (bboxs[i, 3] - bboxs[i, 1]).float() # Height bounding box 
        object_bb_width = (bboxs[i, 2] - bboxs[i, 0]).float() # Width bounding box
        
        if (object_bb_height < 4.0) or (object_bb_width < 4.0): continue 

        # Get the cell row and column index that corresponds to the center of the bounding box 
        cell_row_i = torch.clamp((y_cord_center / yolo_interval).int(), max=num_cells_image_height - 1) 
        cell_col_i = torch.clamp((x_cord_center / yolo_interval).int(), max=num_cells_image_width - 1) 

        # Get the height of the bounding box divided by the actual height of the cell
        bheight = y_cord_center / yolo_interval
        bwidth = x_cord_center / yolo_interval

        # Swap from x,y system to i,j coordinate
        cell_center_i = cell_row_i * yolo_interval + float(yolo_interval) / 2.0
        cell_center_j = cell_col_i * yolo_interval + float(yolo_interval) / 2.0

        # Compute del_x and del_y
        del_x = (x_cord_center.float() - cell_center_j.float()) / yolo_interval
        del_y = (y_cord_center.float() - cell_center_i.float()) / yolo_interval

        # Get the class label
        class_label_of_object = int(labels[i].item())
        if class_label_of_object == 31: continue  # Disregard labels with class label 31
        
        # Get Aspect Ratio
        aspect_ratio = object_bb_height / object_bb_width
        anchor_box_idx = 0
        if aspect_ratio <= 0.2:               anchor_box_idx = 0                                                     ## (45)
        if 0.2 < aspect_ratio <= 0.5:         anchor_box_idx = 1                                                     ## (46)
        if 0.5 < aspect_ratio <= 1.5:         anchor_box_idx = 2                                                     ## (47)
        if 1.5 < aspect_ratio <= 4.0:         anchor_box_idx = 3                                                     ## (48)
        if aspect_ratio > 4.0:                anchor_box_idx = 4
            
        # Create the yolo vector
        # Vector [exsit, x1, y1, height, width, n-encoding]
        yolo_vector = torch.FloatTensor([1, del_x.item(), del_y.item(), bheight.item(), bwidth.item(), 0, 0, 0])
        
        yolo_vector[5 + numericDict[class_label_of_object]] = 1 # One-hot encoding
        
        # Assign to yolo tensor
        yolo_cell_index = cell_row_i.item() * num_cells_image_width + cell_col_i.item()
    
        yolo_tensor[yolo_cell_index, anchor_box_idx] = yolo_vector  # place into proper index

    # Create an augmented yolo tensor
    yolo_tensor_aug = torch.zeros(num_yolo_cells, num_anchor_boxes, yolo_vector_size + 1).float()
    yolo_tensor_aug[:,:,:-1] = yolo_tensor

    # If not exist throw
    for icx in range(num_yolo_cells):
        for iax in range(num_anchor_boxes):
            if(yolo_tensor_aug[icx, iax, 0] == 0):
                yolo_tensor_aug[icx, iax, -1] = 1
    
    return yolo_tensor_aug


# Create DataLoader Code

def ImageProcessing(images, root):
    # Get dir
    image_dir = images["filepath"]

    # Get inputs
    image = Image.open(root + image_dir)

    # Normalize image
    toTensor = tvt.ToTensor()(image)
    toNormalize = tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(toTensor)
    return toNormalize


# Generate Datasets
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imagesDataFrame, root, transform=None):
        super().__init__()
        self.imagesDataFrame = imagesDataFrame
        self.root = root # Directory for folder images

    def bboxAndLabels(self, imageInfo):
        # Split bbox into parts
        x1 = imageInfo["x1"]
        y1 = imageInfo["y1"]
        width = imageInfo["width"]
        height = imageInfo["height"]

        # Get Category
        category = imageInfo["category"]

        labels = torch.zeros(max_num_objects, dtype=torch.uint8) + 31 
        bboxs = torch.zeros(max_num_objects, 4, dtype=torch.uint8)

        # Exract info from dataFrame
        x1_cat = x1[1:-1].split(',')
        y1_cat = y1[1:-1].split(',')
        width_cat = width[1:-1].split(',')
        height_cat = height[1:-1].split(',')
        labelList = category[1:-1].split(',')

        #print(x1_cat, y1_cat, width_cat, height_cat, labelList)
        for j, cat in enumerate(labelList):
            create_box = [float(x1_cat[j]), float(y1_cat[j]), float(width_cat[j]), float(height_cat[j])]

            if (j < max_num_objects):
                bboxs[j] = torch.tensor(create_box, dtype=torch.float)
                labels[j] = int(cat)
        
        return labels, bboxs

    def __len__(self):
        return len(self.imagesDataFrame)

    def __getitem__(self, i):
        imageInfo = self.imagesDataFrame
        imageInfo = imageInfo.iloc[i]
        normalImage = ImageProcessing(imageInfo, self.root)

        # Make YOLO Tensor for processing
        labels, bboxs = self.bboxAndLabels(imageInfo)   
        yoloTensor = createYoloTensor(bboxs, labels)
        
        return normalImage, bboxs, labels, yoloTensor


# Make SkipBlock

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

# Create CNN 

# CNN model
# Based on DLStudio LOADnet2 && notes from class
class NetForYolo(nn.Module):
    # Inspired by Professor Kak's NetForYolo class
    def __init__(self, skip_connections=True, depth=8):
        super(NetForYolo, self).__init__()
        self.skip_connections = skip_connections
        self.depth = depth // 2
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.skip64_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip64_arr.append(Block(in_ch=64, out_ch=64, skip_connections=self.skip_connections))
        
        self.skip64ds = Block(in_ch=64, out_ch=64, downsample=True, skip_connections=self.skip_connections)
        self.skip64to128 = Block(in_ch=64, out_ch=128, skip_connections=self.skip_connections)

        self.skip128_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip128_arr.append(Block(in_ch=128, out_ch=128, skip_connections=self.skip_connections))
        
        self.skip128ds = Block(in_ch=128, out_ch=128, downsample=True, skip_connections=self.skip_connections)
        self.skip128to256 = Block(in_ch=128, out_ch=256, skip_connections=self.skip_connections)
        self.skip256_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip256_arr.append(Block(in_ch=256, out_ch=256, skip_connections=self.skip_connections))
        
        self.skip256ds = Block(in_ch=256, out_ch=256, downsample=True, skip_connections=self.skip_connections)
        
        self.fc_seqn = nn.Sequential(
            nn.Linear(in_features=128*16*16, out_features=6*6*5*9),
                        nn.ReLU(inplace=True),
#                         nn.Linear(in_features=4096, out_features=2048),
#                         nn.ReLU(inplace=True),
                        nn.Linear(in_features=6*6*5*9, out_features=6*6*5*9) # 6x6 grid overlaying the image, 5 anchor boxes, length 9 yolo vector
                        )
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        for i, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            x = skip64(x)
        x = self.skip64ds(x)
        for i, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            x = skip64(x)
        x = self.bn1(x)
        x = self.skip64to128(x)
        for i, skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            x = skip128(x)
        x = self.bn2(x)
        x = self.skip128ds(x)
        x = x.view(-1, 128*16*16)
        x = self.fc_seqn(x)
        return x

# Training Code

# Based on DLStudio run_code_for_training_multi_instance_detection function and notes in class
def train(net, trainLoader, epochs, lr, betas):
    print("Training")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)

    # Criteria
    criterion1 = nn.BCELoss() # If object in image present
    criterion2 = nn.MSELoss() # Regression for bounding box
    criterion3 = nn.CrossEntropyLoss() # One hot encoding of label

    net = net.to(device)
    BCELossIter = []
    MSELossIter = []
    CELossIter = []

    for epoch in range(1, epochs + 1):
        # Initialize Loss Values
        avgBCE_loss, avgMSE_loss, avgCE_loss = 0.0, 0.0, 0.0

        for i, (images, _, _, yolo_tensors) in enumerate(trainLoader):
            images = images.to(device)
            yolo_tensors = yolo_tensors.to(device)

            optimizer.zero_grad()
            
            output = net(images)    
            output = output.view(batch_size, num_yolo_cells, num_anchor_boxes, yolo_vector_size+1)
            
            totalBCE_loss = torch.tensor(0.0, requires_grad=True).float().to(device)
            totalMSE_loss = torch.tensor(0.0, requires_grad=True).float().to(device)
            totalCE_loss = torch.tensor(0.0, requires_grad=True).float().to(device)
        
            # Set yolo_tensor
            yolo_input_tensor = torch.nonzero(output[:, :, :, 0])
            
            # Seperate Values
            batch_axis = yolo_input_tensor[:,0]
            yolo_cells = yolo_input_tensor[:,1]
            anchor_box = yolo_input_tensor[:,2]

            # BCE Loss
            bce_loss = criterion1(nn.Sigmoid()(output[:, :, :, 0]), yolo_tensors[:, :, :, 0])
            totalBCE_loss += bce_loss
            
            # MSE Loss
            predicted_regression_vector = output[batch_axis, yolo_cells, anchor_box, 1:5]
            target_regression_vector = yolo_tensors[batch_axis, yolo_cells, anchor_box, 1:5]
            mse_loss = criterion2(predicted_regression_vector, target_regression_vector)
            totalMSE_loss += mse_loss
            
            # CELoss
            class_probs_vector = output[batch_axis, yolo_cells, anchor_box, 5:-1]
            target_class_vector = torch.argmax(yolo_tensors[batch_axis, yolo_cells, anchor_box, 5:-1], dim=1)
            ce_loss = criterion3(class_probs_vector, target_class_vector)
            totalCE_loss += ce_loss
            
            totalBCE_loss.backward(retain_graph=True)
            totalMSE_loss.backward(retain_graph=True)
            totalCE_loss.backward(retain_graph=True)
            
            optimizer.step()

            avgBCE_loss += totalBCE_loss.item()
            avgMSE_loss += totalMSE_loss.item()
            avgCE_loss += totalCE_loss.item()

            numRunAvg = 25.0
            if (i+1) % numRunAvg == 0: 
                BCELossIter.append(avgBCE_loss / numRunAvg)
                MSELossIter.append(avgMSE_loss / numRunAvg)
                CELossIter.append(avgCE_loss / numRunAvg)

                print("{0} / {1}: {2}, {3}, {4}".format(epoch, epochs, avgBCE_loss, avgMSE_loss, avgCE_loss))  

                avgBCE_loss, avgMSE_loss, avgCE_loss = 0.0, 0.0, 0.0

    return BCELossIter, MSELossIter, CELossIter   


# Plotting Code

def plot_losses(BCEIterloss, MSEIterloss, CEIterloss):
    iterations = range(len(BCEIterloss))
    figure = plt.figure(1)
    plt.plot(iterations, BCEIterloss, label="BCE Loss")
    plt.plot(iterations, MSEIterloss, label="MSE Loss")
    plt.plot(iterations, CEIterloss, label="CE Loss")

    plt.title(f"Loss per Iteration")
    plt.xlabel(f"Iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("train_losses.jpg")

# Skeleton Code from HW5
def display_images(batch_images, batch_labels, batch_bboxs, predicted_labels, predicted_bboxs):
        
    fig, axes = plt.subplots(1)
    batch_images = batch_images /2 + 0.5
    image = np.asarray(tvt.ToPILImage()(batch_images))
    image_gt = np.asarray(tvt.ToPILImage()(batch_images))

    # Labels
    get_cat = [inverse_categories[int(i)] for i in batch_labels if int(i) != 31] # 31
    
    # Bounding Box
    bboxs_in_image = [i for i in batch_bboxs if i.tolist() != [0,0,0,0]]
    
    for j in range(len(bboxs_in_image)):
        # Base
        [x, y, w, h] = batch_bboxs[j]
        image_gt = cv2.rectangle(image_gt, (int(x), int(y)), (int(w), int(h)), color=(0,255,0), thickness=2)
        image_gt = cv2.putText(image_gt, get_cat[j], ((int(x)), (int(y) - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2)

#     axes[0].imshow(image_gt)
    
    for j in range(len(predicted_bboxs)):
        # Predicted
        [x, y, w, h] = predicted_bboxs[j]
        image_pred = cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), color=(0,255,0), thickness=2)
        image_pred = cv2.putText(image_pred, predicted_labels[j], ((int(x)), (int(y) - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2)
        
    axes.imshow(image_pred)
    
    

# Testing Code

def AnalzyePred(pred_regress_vec, icx):
    del_x,del_y = pred_regress_vec[0], pred_regress_vec[1]
    h,w = pred_regress_vec[2], pred_regress_vec[3]
    
    h *= yolo_interval
    w *= yolo_interval       
    
    cell_row_i =  icx // num_cells_image_height
    cell_col_i =  icx % num_cells_image_width
    
    bbox_center_x_coord = cell_col_i * yolo_interval  +  yolo_interval/2  +  del_x * yolo_interval
    bbox_center_y_coord = cell_row_i * yolo_interval  +  yolo_interval/2  +  del_y * yolo_interval
    x1 =  int(bbox_center_x_coord - w / 2.0)
    y1 =  int(bbox_center_y_coord - h / 2.0)
    
    return [x1, y1, int(w + x1), int(h + y1)]

# Based on Professor Kak's run_code_for_testing_multi_instance_detection function
def test(net, valLoader):
    net = net.to(device)

    with torch.no_grad():
        for i, data in enumerate(valLoader):
            batch_images, batch_bbox, batch_labels, batch_yolo_tensor = data
            
            batch_images = batch_images.to(device)
            batch_bbox = batch_bbox.to(device)
            batch_labels = batch_labels.to(device)
            batch_yolo_tensor = batch_yolo_tensor.to(device)
            
            predicted_yolo_tensor = net(batch_images) # Get Prediction
            predicted_yolo_tensor = predicted_yolo_tensor.view(batch_size, num_yolo_cells, num_anchor_boxes, yolo_vector_size+1) # Save YOLO info into tensor

            for ibx in range(predicted_yolo_tensor.shape[0]): # Across batch axis
                
                icx_to_best_anchor_box = {ic: None for ic in range(num_cells_image_height * num_cells_image_width)}
                for icx in range(predicted_yolo_tensor.shape[1]): # Across yolo cell axis
                    cell_pred_i = predicted_yolo_tensor[ibx, icx]
                    prev_best = 0
                    
                    #Compare 
                    for anch_i in range(cell_pred_i.shape[0]):
                        if(cell_pred_i[anch_i][0] > cell_pred_i[prev_best][0]):
                            prev_best = anch_i
                    icx_to_best_anchor_box[icx] = prev_best
                
                # Get the 5 yolo cells
                sorted_icx_to_box = sorted(icx_to_best_anchor_box, 
                                key=lambda x: predicted_yolo_tensor[ibx,x,icx_to_best_anchor_box[x]][0].item(), reverse=True)
                retained_cells = sorted_icx_to_box[:5]
            
                # Identify the objects in the retained cells and extract their bounding boxes
                predicted_bboxs = []
                predicted_labels = []

                for icx in retained_cells:
                    predicted_yolo_vector = predicted_yolo_tensor[ibx, icx, icx_to_best_anchor_box[icx]]
                    target_yolo_vector = batch_yolo_tensor[ibx, icx, icx_to_best_anchor_box[icx]]
                    
                    class_label_predictions = predicted_yolo_vector[-4:]
                    class_labels_probs = torch.nn.Softmax(dim=0)(class_label_predictions)
                    class_labels_probs = class_labels_probs[:-1]
                    if(torch.all(class_labels_probs < 0.2)):
                        predicted_class_label = None

                    else:
                        # Get the predicted class label
                        best_predicted_class_index = (class_labels_probs == class_labels_probs.max())
                        best_predicted_class_index = torch.nonzero(best_predicted_class_index, as_tuple=True)
                        predicted_class_label = class_list[best_predicted_class_index[0].item()]
                        predicted_labels.append(predicted_class_label)

                        # Analyze the predicted regression elements
                        pred_regress_vec = predicted_yolo_vector[1:5].cpu()
                        pred_bb = AnalzyePred(pred_regress_vec, icx)
                        predicted_bboxs.append(pred_bb)
                        
                if(i % 50 == 49):
                    if(predicted_bboxs and predicted_labels):
                        display_images(batch_images[ibx], batch_labels[ibx].tolist(), batch_bbox[ibx],
                                      predicted_labels, predicted_bboxs)

# Run All Code

# Get DataFrame Data
trainDataFrame = pd.read_csv("/scratch/gilbreth/dfarache/ece60146/David/HW6/train_labels.csv")
valDataFrame = pd.read_csv("/scratch/gilbreth/dfarache/ece60146/David/HW6/val_labels.csv")

# Datasets Objects
trainDataset = MyDataset(trainDataFrame, root=root)
valDataset = MyDataset(valDataFrame, root=root)

# Data Loaders
batch_size = 64
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)

## Create neural network
net = NetForYolo(skip_connections=True, depth=8)

# Train Network
BCE_loss, MSE_loss, CE_loss = train(net, trainLoader, epochs=1, lr=1e-5, betas=(0.9, 0.999))

plot_losses(BCE_loss, MSE_loss, CE_loss)

# Test Network
images = test(net, valLoader)


