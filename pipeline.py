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

cap = cv2.VideoCapture('project_video.mp4')

svc,scaler = loadPickledClassifier('trained_model3.pkl')
test_img = cv2.imread('test_images/test5.jpg')
windows_large = slide_window(test_img,y_start_stop=(336,test_img.shape[0]),xy_window=(256,256),xy_overlap=(0.5, 0.5));
windows_medium = slide_window(test_img,y_start_stop=(400,int(test_img.shape[0]-64)),xy_window=(160,160),xy_overlap=(0.5, 0.5));
windows_small = slide_window(test_img,y_start_stop=(400,int(test_img.shape[0]-128)),xy_window=(64,64),xy_overlap=(0.5, 0.5));

#print(start_windows_large)


all_windows = windows_large + windows_medium + windows_small
boxed = draw_boxes(test_img,windows_large,color = (0,0,255))
# boxed = draw_boxes(boxed,windows_large,color = (0,0,255))
# boxed = draw_boxes(boxed,windows_medium_left,color = (255,0,0),thick = 3)
boxed = draw_boxes(boxed,windows_medium,color = (255,0,0),thick = 3)
boxed = draw_boxes(boxed,windows_small,color = (0,255,0),thick = 1)
cv2.imshow('test',boxed)
cv2.waitKey(0)




#print(found_vehicles)
heat = np.zeros(test_img.shape[0:2],dtype='float64')
i = 0
plt.ion()
while cap.isOpened():

    ret,img = cap.read()
    i +=1
    # if i < 250:
    #     continue
    if not ret:
        break
    found_vehicles=  search(test_img,all_windows,svc,scaler)#search_windows(img,all_windows,svc,scaler,color_space,spatial_size,hist_bins,hist_range,orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat);
    found = draw_boxes(test_img,found_vehicles,color = (0,255,0),thick = 1)
    heat = add_heat(heat,found_vehicles)

    heat = apply_threshold(heat,2)
    heat = np.clip(heat, 0, 255)
    # kernel = np.ones((8,8),np.float64)
    # heat = cv2.morphologyEx(np.float64(heat[:,:]), cv2.MORPH_CLOSE, kernel)
    #add more windows in the region of interest of a prossibly matched vehicle
    labels = label(heat)
    # more_windows = []
    # for car_number in range(1, labels[1]+1):
    #     # Find pixels with each car_number label value
    #     nonzero = (labels[0] == car_number).nonzero()
    #     # Identify x and y values of those pixels
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #     # Define a bounding box based on min/max x and y
    #     bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    #     h = (bbox[1][1]-bbox[0][1])
    #     w = (bbox[1][0]-bbox[0][0])
    #     size = max(h,w)
    #     y_start = bbox[0][1]-size
    #     y_end = bbox[1][1]+size
    #     x_start = bbox[0][0]-size
    #     x_end = bbox[1][0]+size
    #
    #     more_windows = slide_window(test_img,x_start_stop=(int(x_start),int(x_end)),y_start_stop=(int(y_start),int(y_end)),xy_window=(int(size),int(size)),xy_overlap=(0.95, 0.95))
    #     found_vehicle=  search_windows(img,more_windows,svc,scaler,color_space,spatial_size,hist_bins,hist_range,orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat);
    #     heat_car = add_heat(np.copy(heat),found_vehicles)
    #     found = draw_boxes(found,found_vehicle,color = (0,0,255),thick = 1)
    #     heat += heat_car
    # heat = apply_threshold(heat,5)
    heat = np.clip(heat, 0, 255)
    cv2.imshow('test',draw_labeled_bboxes(found,labels))
    cv2.waitKey(1)
    #fig = plt.figure()
    # plt.imshow(cv2.resize(heat,(int(img.shape[1]/2),int(img.shape[0]/2))),cmap = 'hot')
    # plt.pause(0.05)
    #fig.tight_layout()
    #plt.show(block = False)
    heat = cool_down(heat,0.2)
