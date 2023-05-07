# %%
# Import all necessary packages
import torch
import torchvision.transforms as tvt
import re

import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
from scipy.stats import wasserstein_distance
import numpy


# %%
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
#os.environ[’PYTHONHASHSEED’] = str(seed)

# %% [markdown]
# # Task 1

# %%
# Import Images 
original_file='/Users/davidfarache/Documents/ECE60146/HW2/StopSignImages/IMG_0174.jpg'
target_file='/Users/davidfarache/Documents/ECE60146/HW2/StopSignImages/IMG_0175.jpg'

# Load images as PIL
original_pil = Image.open(original_file) # load image as PIL
target_pil = Image.open(target_file) # load image as PIL

print(original_pil.size) #Get W x H
print(target_pil.size) #Get W x H

# Convert to tensor and normalize
original_to_tensor = tvt.ToTensor()(original_pil) ## for conversion to [0,1] range
target_to_tensor = tvt.ToTensor()(target_pil) ## for conversion to [0,1] range

# %%
#original_pil.show()
#arget_pil.show()

# %%
bins = 10
original_hist = torch.histc(original_to_tensor, bins=bins, min=-1, max=1)
target_hist = torch.histc(target_to_tensor, bins=bins, min=-1, max=1)

print(original_to_tensor.shape)
print(target_to_tensor.shape)

# %%
#Perform affine transformation
def randAffineTransform(image_to_tensor, degree, trans, scale, shear):
    randAffine = tvt.RandomAffine(degree, translate=trans, scale=scale, shear=shear)
    transformImage = randAffine(image_to_tensor)

    img = tvt.ToPILImage()(transformImage)
    #img.show()
    
    return transformImage

# %%
# Perspective Transform
def persTransform(image_to_tensor, startpoints, endpoints):
    perspective_transformed_image = tvt.functional.perspective(image_to_tensor, startpoints=startpoints, endpoints=endpoints, interpolation=tvt.InterpolationMode.BILINEAR)

    img = tvt.ToPILImage()(transformImage)
    #img.show()
    
    return transformImage

# %%
def computeDistance(original_to_tensor, target_to_tensor, num_bins):
    #create empty tensor to store new RGB normalized data
    histTensorA = torch.zeros( target_to_tensor.shape[0], num_bins, dtype=torch.float )
    histTensorB = torch.zeros( target_to_tensor.shape[0], num_bins, dtype=torch.float )

    # Convert to histogram and normalize
    histsA = [torch.histc(original_to_tensor[ch],bins=num_bins,min=-3.0,max=3.0) for ch in range(3)]
    histsA = [histsA[ch].div(histsA[ch].sum()) for ch in range(3)] ## (13)
    histsB = [torch.histc(target_to_tensor[ch],bins=num_bins,min=-3.0,max=3.0) for ch in range(3)]
    histsB = [histsB[ch].div(histsB[ch].sum()) for ch in range(3)] ## (15)

    # Save into array with idx batch for processing distance
    for ch in range(3):
        histTensorA[ch] = histsA[ch]
        histTensorB[ch] = histsB[ch]

    # Store distance information
    BatchImageRGBDistance = []
    for ch in range(3):
        dist = wasserstein_distance( torch.squeeze( histTensorA[ch] ).cpu().numpy(),
        torch.squeeze( histTensorB[ch] ).cpu().numpy() )
        #print("\n Wasserstein distance for channel %d: " % ch, dist)
        BatchImageRGBDistance.append([ch, dist])
        
    return BatchImageRGBDistance #return distance RGB

# %%
batchRGB = computeDistance(original_to_tensor, target_to_tensor, 10)
print('Red: ' + str(batchRGB[0][1]))
print('Green: ' + str(batchRGB[1][1]))
print('Blue: ' + str(batchRGB[2][1]))

# %%
# Loop for best affine parameters
degParam = [0, 0]
transParam = [1, 1]
scaleParam = [1, 1]
shearParam = [0, 0]
total = 10 ** 12

#Solve for best scale using distance calc
for scale2 in range(1, 10, 1):
    for scale1 in range(scale2, 10, 1):
        transformFig = randAffineTransform(original_to_tensor, (degParam[1], degParam[0]), (transParam[1], transParam[0]), (scale2/10, scale1/10), (shearParam[1], shearParam[0]))
        val = computeDistance(transformFig, target_to_tensor, 10)
        if float(val[0][1] + val[1][1] + val[2][1]) < total:
            total =  float(val[0][1] + val[1][1] + val[2][1])
            scaleParam[0] = scale1/10
            scaleParam[1] = scale2/10

print(scaleParam)
img = tvt.ToPILImage()(transformFig)
img.show()

#Solve for best translation using distance calc
for trans2 in range(0, 10, 1):
    for trans1 in range(trans2+1, 10, 1):
        transformFig = randAffineTransform(original_to_tensor, (degParam[1], degParam[0]), (trans2/10, trans1/10), (scaleParam[1], scaleParam[0]), (shearParam[1], shearParam[0]))
        val = computeDistance(transformFig, target_to_tensor, 10)
        if float(val[0][1] + val[1][1] + val[2][1]) < total:
            total =  float(val[0][1] + val[1][1] + val[2][1])
            transParam[0] = trans1 / 10
            transParam[1] = trans2 / 10

print(transParam)
img = tvt.ToPILImage()(transformFig)
img.show()

#Solve for best degree using distance calc
for shear2 in range(0,720, 10):
    for shear1 in range(shear2,720, 10):
        transformFig = randAffineTransform(original_to_tensor, (degParam[1], degParam[0]), (transParam[1], transParam[0]), (scaleParam[1], scaleParam[0]), (shear2, shear1))
        val = computeDistance(transformFig, target_to_tensor, 10)
        if float(val[0][1] + val[1][1] + val[2][1]) < total:
            total =  float(val[0][1] + val[1][1] + val[2][1])
            shearParam[0] = shear1
            shearParam[1] = shear2

print(shearParam)
img = tvt.ToPILImage()(transformFig)
img.show()

#Solve for best degree using distance calc
for deg2 in range(0, 360*4, 10):
    for deg1 in range(deg2, 360*4, 10):
        transformFig = randAffineTransform(original_to_tensor, (deg2, deg1), (transParam[1], transParam[0]), (scaleParam[1], scaleParam[0]), (shearParam[1], shearParam[0]))
        val = computeDistance(transformFig, target_to_tensor, 10)
        if float(val[0][1] + val[1][1] + val[2][1]) < total:
            total =  float(val[0][1] + val[1][1] + val[2][1])
            degParam[0] = deg1
            degParam[1] = deg2

print(degParam)
img = tvt.ToPILImage()(transformFig)
img.show()


# %%
# Perspective Transformation
endpoints = [[142, 132], [140, 225], [325, 57], [325, 169]]
startpoints = [[110, 120], [105, 225], [268, 125], [369, 231]]

perspectiveTransformeImage = tvt.functional.perspective(original_to_tensor, startpoints=startpoints, endpoints=endpoints, interpolation=tvt.InterpolationMode.BILINEAR)
# hist_perspective_transformed = create_histogram(perspective_transformed_image, bins=10)

tvt.ToPILImage()(perspective_transformed_image).show()

# %% [markdown]
# # Task 2

# %%
import os
import matplotlib.pyplot as plt

# Load files and retirn filename and image
def loadFile(path):
    image = []
    fileName = []
    os.chdir(path)
    for file in os.listdir():
        if '.jpg' in file:
            fileName.append(file)
            image.append(Image.open(file))
                
    return fileName, image

# %%
import torch
class MyDataset(torch.utils.data.Dataset):
    # Obtain meta information (e.g. list of file names) # Initialize data augmentation transforms , etc. pass
                
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.transforms = tvt.Compose([ tvt.RandomAffine(0, scale=(0.7, 1)),
                                        tvt.ColorJitter(brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1)),
                                        tvt.transforms.CenterCrop(1500),
                                        tvt.ToTensor()])
        self.fileName, self.im = loadFile(root)

    def __len__(self):
        return len(self.im)
    # Return the total number of images return 100
    
    def __getitem__(self, index):
        images = self.transforms(self.im[index])
        classLabel = index+1
        return images, classLabel
 
    # Read an image at index and perform augmentations
    # Return the tuple: (augmented tensor, integer label) return torch.rand((3, 256, 256)), random.randint(0, 10)

# %%
# Based on the previous minimal example
my_dataset = MyDataset('/Users/davidfarache/Documents/ECE60146/HW2/MyDatasetFolder/') 
print(len(my_dataset)) # 10
index = 1
print(my_dataset[index][0].shape, my_dataset[index][1]) # torch.Size([3, 256, 256]) 6
index = 5
print(my_dataset[index][0].shape, my_dataset[index][1]) # torch.Size([3, 256, 256]) 8

# %% [markdown]
# # Task 3

# %%
import time

dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4)

# %%
fig, ax = plt.subplots(3, 4)
for batch_index, (images, classLabel) in enumerate(dataloader):
    for i in range(len(classLabel)):
        image = tvt.ToPILImage()(images[i])
        ax[batch_index, i].imshow(image)

        fileName = str(classLabel[i])
        image.save("/Users/davidfarache/Documents/ECE60146/HW2/TrasnformedMyDataset/" + fileName + ".jpeg")

# %%
start_time = time.time()

for i in range(1000):
    num = random.randint(0, 9)
    augmented_image, label = my_dataset[num][0], my_dataset[num][1]

print("Runtime: %s seconds" % (time.time() - start_time))
start_time = time.time()
my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=1000, shuffle=True, num_workers=4)

print("Runtime: %s seconds" % (time.time() - start_time))

# %%


# %%



