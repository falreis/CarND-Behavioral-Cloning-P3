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
* video.mp4 containing the video file from the run test

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

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

**TODO**

##### 1.5 Stabilization

**TODO**


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

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, in the end of the model (model.py lines 83-86). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23-63). 

To test the model, I did some simulations, creating some scenarios to train the model:
- **1 lap** - 1 lap at the track
- **2 laps** - 2 laps at the track
- **3 laps** - 3 laps at the track
- **2 laps reverse** - 2 laps at the simulation in the reverse direction
- **bridge** - cross the bridge simulation
- **reckless** - driving for 2 laps to the board of the road, crossing many times the road without get out of the track

To create the model, I combine all the simulation datas to create a large volume of data samples. I did the following steps to define my best training set. I wanted to use the minimum data set with the best results.

- First, I used a **1 lap training set**. The results were bad;
- Using the **2 laps training set**, the results were so much better;
- Using the **3 laps training set**, the results were as good as using **2 laps training set**, with 30% more cost for training;
- Using the **2 laps training set** with **2 laps reverse**, the results improved a little bit, but not so much. I could reduced the number of trainings from 3 to 2 (not overfitting, but not considerable improvement);
- Adding **bridge training set** I wanted to test if the results were better in the bridge of the circuit. I think that this sample set is useless, because using the **2 laps training set**, the results were the same. I decided to keep this training set even so, to guarantee the good results;
- Using **reckless training set** with combine with **2 laps training set**, the results were worse. The car start to go the the board of the road, without any improvements. In some moment, in a curve, the car went out the track.

The final training set, defined for my model was the combination of the following samples:
- **2 laps**
- **2 laps reverse**
- **bridge**

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

---
### Model Architecture and Training Strategy

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
