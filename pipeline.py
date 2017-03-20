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
cap = cv2.VideoCapture('project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoout = cv2.VideoWriter('output.mov',fourcc, 30.0,( 1280,720))
svc,scaler = loadPickledClassifier('trained_model3.pkl')
test_img = cv2.imread('test_images/test5.jpg')
windows_large = slide_window(test_img,y_start_stop=(336,test_img.shape[0]),xy_window=(256,256),xy_overlap=(0.5, 0.5));
windows_medium = slide_window(test_img,y_start_stop=(400,int(test_img.shape[0]-64)),xy_window=(160,160),xy_overlap=(0.5, 0.5));
windows_small = slide_window(test_img,y_start_stop=(400,int(test_img.shape[0]-128)),xy_window=(64,64),xy_overlap=(0.5, 0.5));

#print(start_windows_large)


# all_windows = windows_large + windows_medium + windows_small
# boxed = draw_boxes(test_img,windows_large,color = (0,0,255))
# # boxed = draw_boxes(boxed,windows_large,color = (0,0,255))
# # boxed = draw_boxes(boxed,windows_medium_left,color = (255,0,0),thick = 3)
# boxed = draw_boxes(boxed,windows_medium,color = (255,0,0),thick = 3)
# boxed = draw_boxes(boxed,windows_small,color = (0,255,0),thick = 1)
# cv2.imshow('test',boxed)
# cv2.waitKey(0)


def findOccludedVehicles(cars,car):
    occluded = []
    for c in cars:
        if c != car:
            dist = np.linalg.norm(c.centr.distance - car.centr.distance)
            if dist <  car.getLength()/2 + c.getLength()/2 +32:
                occluded.append(c)
    return occluded



#print(found_vehicles)
heat = np.zeros(test_img.shape[0:2],dtype='float64')
i = 0
plt.ion()
cars = []
while cap.isOpened():

    ret,img = cap.read()
    i +=1
    # if i < 220:
    #     continue
    if not ret:
        break
    found_vehicles=  search(img,svc,scaler)#search_windows(img,all_windows,svc,scaler,color_space,spatial_size,hist_bins,hist_range,orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat);
    # found = draw_boxes(img,found_vehicles,color = (0,255,0),thick = 1)
    new_heat = np.zeros(test_img.shape[0:2],dtype='float64')
    new_heat = add_heat(new_heat,found_vehicles)
    new_heat = cv2.blur(new_heat,(8,8))
    heat = np.clip((heat + new_heat*2)/3, 0, 255)
    print(np.max(heat))
    # kernel = np.ones((8,8),np.float64)
    # heat = cv2.morphologyEx(np.float64(heat[:,:]), cv2.MORPH_CLOSE, kernel)
    #add more windows in the region of interest of a prossibly matched vehicle
    thresh = apply_threshold(heat,4 )

    labels = label(thresh)
    bboxes = get_labeled_boxes(labels)
    for box in bboxes:
        matched = False
        for car in cars:
            if car.matchbbox(box):
                car.appendBox(box)
                #look for cars that are potentially occluded by this vehicle
                # occluded = findOccludedVehicles(cars,car)
                # for c in occluded:
                #     if np.linalg.norm(c.centr.distance - car.centr.distance) > 1 and np.linalg.norm(c.size.distance - car.size.distance) > 1:
                #         c.appendBox(box)#Make same reading count for all occluded vehicles
                #     else:
                #         cars.remove(c)
                matched = True
                break;
        if not matched:
            b = np.array(box)
            b = b[1]-b[0]
            if b[0]*b[1] > 2730:
                # print("new car")
                # input()
                car = Car(bbox = box,timestamp = i-1)
                car.appendBox(box)
                cars.append(car)
    drawboxes = []
    for car in cars:
        car.update(i)
        box = car.getBbox()
        if not box is None:
            drawboxes.append(box)
        if car.centr.isreset:
            cars.remove(car)
    print(drawboxes)
    print('cars length',len(cars))
    out_img = draw_boxes(np.copy(img),drawboxes,color = (255,0,0),thick = 6)
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
    #     heat_car = add_heat(np.copy(heat),found_vehicles)q
    #     found = draw_boxes(found,found_vehicle,color = (0,0,255),thick = 1)
    #     heat += heat_car
    # heat = apply_threshold(heat,5)
    #out_img = draw_labeled_bboxes(img,labels)
    cv2.imshow('test',out_img)
    videoout.write(out_img)
    showHeat = np.uint8(heat/np.max(heat)*255)
    showHeat = cv2.applyColorMap(showHeat, cv2.COLORMAP_HOT)
    cv2.imshow('heat',cv2.resize(showHeat,(int(img.shape[1]/2),int(img.shape[0]/2))))
    cv2.waitKey(1)

    #fig = plt.figure()
    # plt.imshow(cv2.resize(heat,(int(img.shape[1]/2),int(img.shape[0]/2))),cmap = 'hot')
    # plt.pause(0.05)
    #fig.tight_layout()
    #plt.show(block = False)
    #heat = cool_down(heat,.1)
cap.release()
videoout.release()
