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
from sklearn.externals import joblib
import glob

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

def loadFeatures():
    vehicleFiles = [f for f in glob.glob(vehiclesPath + '/**/*.png', recursive=True)]
    nonVehicleFiles = [f for f in glob.glob(nonVehiclesPath + '/**/*.png', recursive=True)]
    print('vehicles',len(vehicleFiles))
    print('non-vehicles',len(nonVehicleFiles))

    #store vehicle features
    car_features = []
    notcar_features = []



    for i in range(len(vehicleFiles)):
        img = cv2.imread(vehicleFiles[i])
        imfeatures = extract_features(img,color_space,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat)
        car_features.append(imfeatures)
    for i in range(len(nonVehicleFiles)):
        img = cv2.imread(nonVehicleFiles[i])
        imfeatures = extract_features(img,color_space,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat)
        notcar_features.append(imfeatures)
    print(car_features[0].shape)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    return X_train, X_test,y_train,y_test,X_scaler


if __name__ == "__main__":
    t=time.time()
    X_train, X_test,y_train,y_test,X_scaler = loadFeatures()
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to load Features...')
    svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC on Train Data= ', round(svc.score(X_train, y_train), 4))
    print('Test Accuracy of SVC on Test Data= ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    joblib.dump(X_scaler,'scaler.pkl')
    joblib.dump(svc,'trained_model3.pkl')


def loadPickledClassifier(classifier):
    svc = joblib.load(classifier)
    scaler = joblib.load('scaler.pkl')
    return svc,scaler
def search(img,windows,svc,scaler):
    scales = [1.5,1]
    start = [400,400,400]
    stop = [656,img.shape[0],img.shape[0]]
    found = []
    for i in range(len(scales)):
        print(i)
        found = found + find_cars(img,start[i],stop[i],scales[i],svc,scaler,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,spatial_feat,hist_feat,hog_feat)
    print(found)
    return found
