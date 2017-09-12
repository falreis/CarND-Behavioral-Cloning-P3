[//]: # (Image References)
[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "NVidia Network Architecture"

## Behavioral Cloning

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 containing the output video file

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### 4. Important

**All the data set was zipped to reduce the number of images files in Github repository**

To unzip the files, follow the commands below, in each data directory:
```sh
zip -FF IMG.zip --out full.zip
unzip full.zip
```

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

##### 1.1 First Architecture

In the beggining of the tests, I used the LeNet architecture and it had some problems to stabilize the vehicle. Using the Nvidia architecture, the problems reduced a lot and I decided to keep the architecture.

##### 1.2 Final Architecture

My model follow the architecture developed by the autonomous vehicle team in Nvidia, as the image above.

![ CNN architecture. The network has about 27 million connections and 250 thousand parameters (Credits: Nvidia Parallel for all).][image1]
(Credits: Nvidia - Parallel for All)

The differences between my architecture and the NVidia architecture is the images shapes. My shapes uses the images sizes from the simulator (160x320x3). I cropped the images to the shape 65x320x3 (70 pixels on the top and 25 at the bottom). 

The cropped images and the architecture followed the video released at class from Udacity and I didn't do any improvements. 

##### 1.3 Normalization

For normalize the data, I started with the simple normalization *(pixel / 255) - 0.5*. Trying to improve my model, I used the OpenCV normalization function in the images from the simulation. The results were worse then the simple normalization with worse cpu/gpu proccess time performance. Then I decided to step back to the simple normalization.
```sh
(pixel / 255) - 0.5
```

##### 1.4 Steering angles

To increase the performance of the solution, it was used the complementary images from the simulation. The left and the right images increased the knowledges of the neural network. To use the complementary images, was necessary to correct the mesurements of the datalog, to keep the vehicle on track.

##### 1.5 Stabilization

Making some tests in the solution, I discovered that I could make some improvements if I change a little bit the mesurements values for all data. If I multiply/divide the value of the mesurements (left, right) for a small parameter (1.1 or 1.2), the system predicted the changes faster and get more stable on the track.

I used this idea to improve my model and refine my algorithm. The value of all the mesurements was multiplied/divided by 1.1, increasing the system response (model.py lines 16, 37-40).


#### 2. Detailed Model Architecture

The detailed architecture is detailed bellow. 

| Layer | Description | 
|:-----:|:-----------:| 
| **1** ||
| Input | 160x320x3 RGB Image | 
| Crop | 65x320x3 RGB Image |
| Normalization | *(pixel / 255) - 0.5* |
|||
| **2** ||
| Convolution 5x5 | Filter number 24, kernel size 5x5, subsample 2x2
| RELU | |
|||
| **3** ||
| Convolution 5x5 | Filter number 36, subsample 2x2
| RELU | |
|||
| **4** ||
| Convolution 5x5 | Filter number 48, subsample 2x2
| RELU | |
|||
| **5** ||
| Convolution 3x3 | Filter number 64
| RELU | |
|||
| **6** ||
| Convolution 3x3 | Filter number 64
| RELU | |
|||
| **7** ||
| Flatten | - |
|||
| **8** ||
| Dense | 100 (Fully Connected Layer) |
|||
| **9** ||
| Dense | 50 (Fully Connected Layer) |
|||
| **10** ||
| Dense | 10 (Fully Connected Layer) |
|||
| **11** ||
| Dense | 1 (Output) |
|||

The model contains dropout layers in order to reduce overfitting, in the end of the model (model.py lines 83-86). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 23-63). 

#### 3. Training

##### 3.1 Collecting Data & Training Sets

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Then I started to do some others scenarios, to use to train the model. The scenarios that I used is described below.

- **1 lap** - 1 lap at the track (on the center lane driving)
- **2 laps** - 2 laps at the track (on the center lane driving)
- **3 laps** - 3 laps at the track (on the center lane driving)
- **2 laps reverse** - 2 laps at the simulation in the reverse direction
- **bridge** - cross the bridge simulation
- **reckless** - driving for 2 laps to the board of the road, crossing many times the road without get out of the track

##### 3.2 Training Process

To create the model, I combine all the simulation datas to create a large volume of data samples. I did the following steps to define my best training set. I wanted to use the minimum data set with the best results.

- First, I used a **1 lap training set**. The results were bad;
- Using the **2 laps training set**, the results were so much better;
- Using the **3 laps training set**, the results were as good as using **2 laps training set**, with 30% more cost for training;
- Using the **2 laps training set** with **2 laps reverse**, the results improved a little bit, but not so much. I could reduced the number of trainings from 3 to 2 (not overfitting, but not considerable improvement);
- Adding **bridge training set** I wanted to test if the results were better in the bridge of the circuit. I think that this sample set is useless, because using the **2 laps training set**, the results were the same. I decided to keep this training set even so, to guarantee the good results;
- Using **reckless training set** with combine with **2 laps training set**, the results were worse. The car start to go the the board of the road, without any improvements. In some moment, in a curve, the car went out the track.

##### 3.3 Appropriate Training Data

The final training set, defined for my model was the combination of the following samples:
- **2 laps**
- **2 laps reverse**
- **bridge**

The model was trained for 2 epochs. Train for 3 epochs make sometimes made few improvements and some made worse results then 2 epochs. The increase of the number of trainings causes overfitting.

##### 3.4 Training and Validation Data

After the collection process and the definition of the appropriate training data, I had 14895 data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. Then I had 11916 train samples, and 2979 validade samples.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

Some improvements also were made to try to increase the performance of the system. The first tuned parameter was the regulation to correct the mesurement for the left and the right images. The parameter was defined as 0.22, added or subtract in the side images.

One other tuned parameter that I used but had the opposite effect that I expected was to flip the images. Flipping the images, the model become slowly without any great improvemtns. In the end, I decided to not flip the images and only use my training sets.


