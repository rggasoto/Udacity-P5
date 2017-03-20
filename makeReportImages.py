import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from imgUtils import *
from trainClassifier import *
from scipy.ndimage.measurements import label
from Car import Car
from glob import glob
import random

vehiclesPath = 'vehicles'
nonVehiclesPath = 'non-vehicles'

#Select which features to use
spatial_feat=True
hist_feat=True
hog_feat=True
#Features parameters
color_space='YCrCb'
spatial_size=(32, 32)
hist_bins=32
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'#'GRAY'
hist_range=(0,256)
svc,scaler = loadPickledClassifier('trained_model3.pkl')


test_imgs = [f for f in glob('test_images/*.jpg')]
vehicleFiles = [f for f in glob(vehiclesPath + '/**/*.png', recursive=True)]
nonVehicleFiles = [f for f in glob(nonVehiclesPath + '/**/*.png', recursive=True)]

##Features Analysis
vehicleFig = cv2.imread(vehicleFiles[int(random.random()*len(vehicleFiles))])
nonVehicleFig = cv2.imread(nonVehicleFiles[int(random.random()*len(nonVehicleFiles))])
hogv = cv2.cvtColor(vehicleFig,cv2.COLOR_BGR2YCrCb)
hognv = cv2.cvtColor(nonVehicleFig,cv2.COLOR_BGR2YCrCb)

vehicleHist = color_hist(hogv,hist_bins)
nonVehicleHist = color_hist(hognv,hist_bins)
hogv = cv2.cvtColor(vehicleFig,cv2.COLOR_BGR2YCrCb)
hognv = cv2.cvtColor(nonVehicleFig,cv2.COLOR_BGR2YCrCb)
hog_vehicle = np.zeros_like(hogv)
hog_non_vehicle = np.zeros_like(hogv)
for i in range(3):
    _,hog_vehicle[:,:,i] = get_hog_features(hogv[:,:,i], orient, pix_per_cell, cell_per_block,
                            vis=True, feature_vec=False)
    _,hog_non_vehicle[:,:,i] = get_hog_features(hognv[:,:,i], orient, pix_per_cell, cell_per_block,
                            vis=True, feature_vec=False)
plt.figure()
plt.subplot(2,3,1)
img_show = cv2.cvtColor(vehicleFig,cv2.COLOR_BGR2RGB)
plt.imshow(img_show)
plt.subplot(2,3,2)
for i in range(3):
    x = np.arange(hist_bins)+hist_bins*i
    print(len(vehicleHist[i:hist_bins*(i+1)]))
    color = [0,0,0,1]
    color[i] = 1
    plt.bar(x,height=vehicleHist[hist_bins*i:hist_bins*(i+1)],color = color,width=1)
plt.subplot(2,3,3)
plt.imshow(hog_vehicle[:,:,0],cmap='gray')
plt.subplot(2,3,4)
img_show = cv2.cvtColor(nonVehicleFig,cv2.COLOR_BGR2RGB)
plt.imshow(img_show)
plt.subplot(2,3,5)
for i in range(3):
    x = np.arange(hist_bins)+hist_bins*i
    print(len(vehicleHist[i:hist_bins*(i+1)]))
    color = [0,0,0,1]
    color[i] = 1
    plt.bar(x,height=nonVehicleHist[hist_bins*i:hist_bins*(i+1)],color = color,width=1)
plt.subplot(2,3,6)
plt.imshow(hog_non_vehicle[:,:,0],cmap='gray')
plt.show()


##Sliding windows

img = cv2.imread(test_imgs[0])
## Draw sliding window boxes:
imgBoxed = np.copy(img)
scales = [1.5*64,1*64,0.8*64]
start = [400,400,400]
stop = [656,img.shape[0]-160,500]

for i in range(len(scales)):
    color = [0,0,0]
    color[i] = 255
    color = tuple(color)
    cv2.rectangle(imgBoxed,(0,start[i]),(img.shape[1],stop[i]),color,6)
    xend = 0
    yend = start[i]
    while xend < img.shape[1]:
        yend=start[i]
        while yend < stop[i]:
            cv2.rectangle(imgBoxed,(int(xend),int(yend)),(int(xend+scales[i]),min(int(yend+scales[i]),stop[i])),color,2)
            yend +=int(scales[i])
        xend+=int(scales[i])
    xend = 0
    yend = start[i]
    while xend < img.shape[1]:
        yend=start[i]
        while yend < stop[i]:
            cv2.rectangle(imgBoxed,(int(xend),int(yend)),(int(xend+scales[i]/4),min(int(yend+scales[i]/4),stop[i])),color,1)
            yend +=int(scales[i]/4)
        xend+=int(scales[i]/4)

cv2.imwrite('sliding_windows.png',imgBoxed)

found_vehicles=  search(img,svc,scaler)
new_heat = np.zeros(img.shape[0:2],dtype='float64')
new_heat = add_heat(new_heat,found_vehicles)
# new_heat = cv2.blur(new_heat,(32,32))
heat = np.clip(new_heat, 0, 255)
thresh = apply_threshold(np.copy(heat),4 )
showHeat = np.uint8(heat/np.max(thresh)*255)
showHeat = cv2.applyColorMap(showHeat, cv2.COLORMAP_HOT)
labels = label(thresh)
cars_found = draw_labeled_bboxes(img,labels)
cv2.imshow('boxed',imgBoxed)
cv2.imshow('heat',showHeat)
cv2.imshow('cars',cars_found)
cv2.imwrite('carsfound.png',cars_found)
cv2.imwrite('heatmap.png',showHeat)
cv2.waitKey(0)
