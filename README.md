
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hist_hog.png
[image2]: ./output_images/sliding_windows.png
[image3]: ./output_images/heatmap.png
[image4]: ./output_images/carsfound.png
[video1]: ./project_video.mp4


##Extracting Features from the dataset

The feature extraction and training of the classifier are available in the code file `trainClassifier.py`, which contains the classifier configuration.

The dataset was divided in  `vehicle` and `non-vehicle` images.  The images are then passed in a feature extractor, that extracts spatial features, the histogram on `YCrCb` channel, and the HOG features.

The miage below show a comparison between a vehicle and a non-vehicle image. It can be seen that while the histogram have some differences, specially in Y channel, the major change comes from the HOG, that contains significantly different features between vehicles and non-vehicles.
![alt text][image1]

####2. Parameters Configuration

The parameters were configured using experimental analysis by verifying the final accuracy on the test images. As a cross-section of 20% from the dataset was used to evaluate the classifier performance, the test was performed 5 times in each of the selected configurations. The spatial size and number of bins for the histogram were set fixed to (32,32) and 32, respectively, while the HOG features were configured as follows:

|Configuration | orientations| Pix per cell|cells per block| channel |
|:---:|:---:|:---:|:---:|:---:|
|1 |9|8|2|All|
|2 |8|8|2|All|
|3 |9|8|2|Y  |
|4 |8|16|2|All|
|5 |12|8|2|Y|

Out of all configurations, the one that performed better with reasonable computing time was configuration 1, which was selected for the remainder of the project.


####3. Classifier Training

The classifier of choice was a Linear SVM. The parameters used for the training were Penalty C = 1.0, tolerance of 0.0001, and internal scaling and normalizer for the classifier (meaning the images don't need to be normalized and zero mean, as sklearn handles that internally).

###Vehicle Search

In order to search for the vehicle, one could pick sub-images from the frame and check whether the vehicle is there. However, when the windows overlap this technique is inefficient as many of the features on the overlapping areas are computed more than once. This is of particular importance as extracting the HOG features is an expensive procedure.
Instead, what was done was compute one histogram for the entire search region, and then select the features in the sub-image region out of the HOG. In order to search windows larger or smaller than the original 64x64 trained image, the frame is scaled down or up, in order to accommodate for the new image size searched.

In the pipeline, three regions of search were used, one with scales `[1.5,1.,0.8]`. The overlapping was done by advancing one cell in the HOG space, resulting in 75% overlapping in all regions. Thinner line represents where each window starts, medium lines give a sense on the size of each scale, while the outer bounds is the complete searched space.


![alt text][image2]


---

### Video Implementation

Here's a [link to the video](./output.mov)


#### Filtering out bad readings.
  Although the classifier passed with 99.3% accuracy in the dataset, it still collects a reasonable amount of noise when applied to the video. in order to filter this noise out, a heat map was applied to the overlapping measurements. Whenever 4 overlapping measurements collect a vehicle, it will be passed on a persistence filter, that only allows it to display after seen for some frames. the same persistence model only allows a measurement to leave screen after it is not detected for a certain amount of frames too.
  This effectively removed the noise on the video, leaving only vehicle measurements.

Here's an image of the heatmap, and the detected vehicles obtained from it
![alt text][image3]
![alt text][image4]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
