# %% [markdown]
# # Library

# %%
%matplotlib inline
from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import skimage
import cv2
import pandas as pd

import matplotlib.pyplot as plt
import pylab
import random
from PIL import Image
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# %% [markdown]
# # Display Images

# %%
# Based from skeleton code given
def display_random_image_with_bbox(new_image_path, filename, bbox, cat):
    file = new_image_path + '/' + filename
    image = Image.open(file)
    [x, y, w, h] = bbox
    fig, ax = plt.subplots(1,1)
    image = np.uint8(image)
    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(36, 255, 12), thickness=2)
    image = cv2.putText(image, cat, (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)

    ax.imshow(image)
    ax.set_axis_off()
    plt.axis("tight")
    plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10), dpi = 150)
    axs = axs.flatten()
    axs_count = 0   

# %%
def display_images(saveForPlotting):
    fig, ax = plt.subplots(3, 3)
    row, col = 0, 0
    
    #print(len(saveForPlotting))
    for arr in saveForPlotting:
        file = arr[0] + '/' + arr[1]
        image = Image.open(file)
        
        [x, y, w, h] = arr[2]
        image = np.uint8(image)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(36, 255, 12), thickness=2)
        image = cv2.putText(image, arr[3], (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)
        ax[row, col].imshow(image)

        # Increment through row
        col += 1

        if col == 3: 
            col = 0
            row += 1
            
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10), dpi = 150)
    plt.show()
    axs = axs.flatten()

# %% [markdown]
# # Adjust and Change Images

# %%
def resizeBBOX(bbox, startWidth, startHeight, resize):
        x_scale = resize / startWidth
        y_scale = resize / startHeight
        bboxResize = np.zeros(4)

        bboxResize[0] = int(x_scale * bbox[0])
        bboxResize[1] = int(y_scale * bbox[1])
        bboxResize[2] = int(x_scale * bbox[2])
        bboxResize[3] = int(y_scale * bbox[3])
        
        return bboxResize.tolist()

# %%
def resizeAndRepairImage(start_image_path, new_image_path, filename, bbox, width, height, resize):
        img = Image.open(start_image_path + '/' + filename)
        width, height = img.size

        image = img.convert(mode="RGB")

        img = image.resize((resize, resize), Image.BOX)
        if '\ 2' in filename:
                print('f')
        img.save(new_image_path + '/' + filename)

# %%
def saveAsDataFrame(ids, cats, filepaths, x1, y1, width, height, type):
    # Containers for data to go in training label csv
    columns = ["id", "category", "filepath", "x1", "y1" "width", "height"]
    dataFrame = pd.DataFrame(columns=columns)
    dataFrame["id"] = ids
    dataFrame["category"] = cats
    dataFrame["filepath"] = filepaths
    dataFrame["x1"] = x1
    dataFrame["y1"] = y1
    dataFrame["width"] = width
    dataFrame["height"] = height  
    dataFrame.to_csv("{0}_labels.csv".format(type))
    

# %% [markdown]
# # Choose Images Within Parameters

# %%
# Get Random Images from set
def ImageSelection(start_image_path, new_image_path, cocoObj, class_list, type, spotSave):
    # Save File Location List
    saveImportant = []
    saveForPlotting = []

    # Save image info
    ids = []
    cats = []
    filepaths = []
    x1 = []
    y1 = []
    widthSave = []
    heightSave = []

    for cat in class_list:
        # get all images containing given categories
        catIds = cocoObj.getCatIds(catNms=[cat]) # Get ids from annotations
        imgIds = cocoObj.getImgIds(catIds=catIds ) # Load images ids of chosen annotations ids
        img = cocoObj.loadImgs(ids=imgIds) # Get images
        numPlots = 0 # Make X number plots
        

        #Loop per image
        for idx, images  in enumerate(img):
            annIds = cocoObj.getAnnIds(imgIds=images['id'], catIds=catIds, iscrowd=False) # Get dictionary value
            anns = cocoObj.loadAnns(annIds) # Get annotations
            domObj = 0 # Amount of Dominante Obj
            spot = 0
            for jdx, ann in enumerate(anns):
                if ann['area'] > 200 * 200: #check if means parameter of dominate obj
                    domObj += 1
                    spot = jdx


            if domObj == 1: # only one dom allowed
                # Resize Images
                width, height = images['width'], images['height']
                boxDict = []
                #for ann in anns:
                bboxResize = resizeBBOX(anns[jdx]['bbox'], int(width), int(height), resize=256) #Adjust Box
                resizeAndRepairImage(start_image_path, new_image_path, images['file_name'], anns[jdx]['bbox'], width, height, resize=256) # Save Adjust Image 

                ids.append(images['id'])
                cats.append(cat)
                filepaths.append("{0}/{1}".format(spotSave,images['file_name']))
                x1.append(bboxResize[0])
                y1.append(bboxResize[1])
                widthSave.append(bboxResize[2])
                heightSave.append(bboxResize[3])

                if numPlots < 3:
                    #display_random_image_with_bbox(new_image_path,images['file_name'], bboxResize, cat)
                    saveForPlotting.append([new_image_path, images['file_name'], bboxResize, cat])
                    print(saveForPlotting)
                    numPlots += 1

    display_images(saveForPlotting)       
    saveAsDataFrame(ids, cats, filepaths, x1, y1, widthSave, heightSave, type)

# %%
# Input
train_json = '/Users/davidfarache/Documents/ECE60146/HW5/annotations/instances_train2014.json'
val_json = '/Users/davidfarache/Documents/ECE60146/HW5/annotations/instances_val2014.json'

train_path = '/Users/davidfarache/Documents/ECE60146/HW5/train2014'
train_data_path = '/Users/davidfarache/Documents/ECE60146/HW5/trainingData'

val_path = '/Users/davidfarache/Documents/ECE60146/HW5/val2014'
val_data_path = '/Users/davidfarache/Documents/ECE60146/HW5/valData'

trainSaveSpot = 'trainingData'
valSaveSpot = 'valData'

class_list = ['pizza', 'bus', 'cat']

# %%
cocoTrain = COCO(train_json)
cocoVal = COCO(val_json)

ImageSelection(train_path, train_data_path, cocoTrain, class_list, 'train', trainSaveSpot)
ImageSelection(val_path, val_data_path, cocoVal, class_list, 'val', valSaveSpot)

# %%



