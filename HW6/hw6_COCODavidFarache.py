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
inverse_categories = {6: "bus", 17: "cat", 59: "pizza"}

# %%
# Based from skeleton code given
def display_random_image_with_bbox(image_path, bboxs, cat_set):
    file = image_path
    image = Image.open(file)

    print(bboxs)   
    print(cat_set) 
    for bbox, cat in zip(bboxs, cat_set):
        [x, y, w, h] = bbox
        fig, ax = plt.subplots(1,1)
        image = np.uint8(image)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(36, 255, 12), thickness=2)
        image = cv2.putText(image, inverse_categories[cat], (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)

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
        file = arr[0]
        image = Image.open(file)
        print(arr)
        for cat, bbox in zip(arr[2], arr[1]):
            [x, y, w, h] = bbox
            image = np.uint8(image)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(36, 255, 12), thickness=2)
            image = cv2.putText(image, inverse_categories[cat], (int(x), int(y - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)
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
def resizeAndRepairImage(start_image_path, new_image_path, filename, resize):
        img = Image.open(start_image_path + '/' + filename)
        width, height = img.size

        image = img.convert(mode="RGB")

        img = image.resize((resize, resize), Image.BOX)
        img.save(new_image_path + '/' + filename)

# %%
def saveAsDataFrame(ids, cats, filepaths, x1, y1, width, height, type):
    # Containers for data to go in training label csv
    columns = ["id", "category", "filepath", "x1", "y1", "width", "height"]
    dataFrame = pd.DataFrame(columns=columns)
    dataFrame = dataFrame.astype('object')

    dataFrame["id"] = np.array(ids)
    dataFrame["category"] = cats
    dataFrame["filepath"] = filepaths
    dataFrame["x1"]  = np.array(x1)
    dataFrame["y1"]  = y1
    dataFrame["width"]  = width
    dataFrame["height"]  = height

    dataFrame.to_csv("{0}_labels.csv".format(type))

    return dataFrame
    

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
    widthFrame = []
    heightFrame = []
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

            # Saving List
            id_set = []
            cat_set = []
            x1_set = []
            y1_set = []
            widthFrame_set = []
            heightFrame_set = []
            bbox_set = []

            forGroundImge = 0
            for jdx, ann in enumerate(anns):
                
                width, height = images['width'], images['height']

                if ann['area'] > 64 * 64: #check if means parameter of dominate obj                    
                    # Resize Images                    
                    bboxResize = resizeBBOX(ann['bbox'], int(width), int(height), resize=256) #Adjust Box

                    # Append info
                    id_set.append(ann['id'])
                    cat_set.append(ann['category_id'])
                    x1_set.append(bboxResize[0])
                    y1_set.append(bboxResize[1])
                    widthFrame_set.append(bboxResize[2])
                    heightFrame_set.append(bboxResize[3])
                    bbox_set.append(bboxResize)

                    forGroundImge = 1

            if forGroundImge == 1:
                resizeAndRepairImage(start_image_path, new_image_path, images['file_name'], resize=256) # Save Adjust Image 

                cats.append(cat_set)
                filepaths.append("{0}/{1}".format(spotSave,images['file_name']))
                ids.append(id_set)
                x1.append(x1_set)
                y1.append(y1_set)
                widthFrame.append(widthFrame_set)
                heightFrame.append(heightFrame_set)
                
                if numPlots < 3 and len(cat_set) > 1:
                    #display_random_image_with_bbox("{0}/{1}".format(spotSave,images['file_name']), bbox_set, cat_set)
                    saveForPlotting.append(["{0}/{1}".format(spotSave,images['file_name']), bbox_set, cat_set])
                    numPlots += 1

    display_images(saveForPlotting)
    dataFrame = saveAsDataFrame(ids, cats, filepaths, x1, y1, widthFrame, heightFrame, type)
    return dataFrame

# %%
# Input
train_json = '/Users/davidfarache/Documents/ECE60146/HW6/annotations/instances_train2014.json'
val_json = '/Users/davidfarache/Documents/ECE60146/HW6/annotations/instances_val2014.json'

train_path = '/Users/davidfarache/Documents/ECE60146/HW6/train2014'
train_data_path = '/Users/davidfarache/Documents/ECE60146/HW6/trainingData'

val_path = '/Users/davidfarache/Documents/ECE60146/HW6/val2014'
val_data_path = '/Users/davidfarache/Documents/ECE60146/HW6/valData'

trainSaveSpot = 'trainingData'
valSaveSpot = 'valData'

class_list = ['pizza', 'bus', 'cat']

# %%
cocoTrain = COCO(train_json)
cocoVal = COCO(val_json)

trainDataFrame = ImageSelection(train_path, train_data_path, cocoTrain, class_list, 'train', trainSaveSpot)
valDataFrame = ImageSelection(val_path, val_data_path, cocoVal, class_list, 'val', valSaveSpot)

print(len(trainDataFrame))
print(len(valDataFrame))

# %%
print(trainDataFrame.columns)
print(valDataFrame.columns)

# %%
#print(trainDataFrame['category'])
saving_set = []
for i, val in enumerate(trainDataFrame['category']):
    if val[0] not in saving_set:
        print(trainDataFrame['filepath'].iloc[i])
        saving_set.append(val[0])
        print(val[0])

# %%



