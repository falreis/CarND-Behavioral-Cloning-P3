[//]: # (Image References)
[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "NVidia Network Architecture"
[image2]: ./writeup_images/2laps_center_camera.jpg "Center lane driving - center camera"
[image3]: ./writeup_images/2laps_left_camera.jpg "Center lane driving - left camera"
[image4]: ./writeup_images/2laps_right_camera.jpg "Center lane driving - right camera"
[image5]: ./writeup_images/bridge_center_camera.jpg "Bridge simulation - center camera"
[image6]: ./writeup_images/bridge_left_camera.jpg "Bridge simulation - left camera"
[image7]: ./writeup_images/bridge_right_camera.jpg "Bridge simulation - right camera"


## Behavioral Cloning

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

##### 1.1 First Architecture

In the beggining of the tests, I used the LeNet architecture and it had some problems to stabilize the vehicle. Using the Nvidia architecture, the problems reduced a lot and I decided to keep the architecture.

##### 1.2 Final Architecture

My model follow the architecture developed by the autonomous vehicle team in Nvidia, as the image above.

![ CNN architecture. The network has about 27 million connections and 250 thousand parameters (Credits: Nvidia Parallel for all).][image1]
(Credits: Nvidia - Parallel for All)

The differences between my architecture and the NVidia architecture is the images shapes (the simulator return images in the shape 160x320x3). I cropped the images to the shape 65x320x3 (70 pixels cropped on the top and 25 pixels cropped at the bottom). I add some dropouts layers to prevent overfitting.

##### 1.3 Normalization

To normalize the data, I started with the simple normalization *(pixel / 255) - 0.5*. Trying to improve my model, I used the OpenCV normalization function in the images from the simulation. The results were worse then the simple normalization with worse cpu/gpu proccess time performance. Then I decided to step back to the simple normalization.
```sh
(pixel / 255) - 0.5
```

##### 1.4 Steering angles

To increase the performance of the solution, it used complementary images. The left and the right images increased the knowledges of the neural network. It was necessary to correct the mesurements of the datalog, add some correction value to the left and right image, to keep the vehicle on track.

##### 1.5 Stabilization

Making some tests in the solution, I discovered that I could make some improvements if I change a little bit the mesurements values for all data. If I multiply/divide the value of the mesurements (turn left or turn right) for a small parameter (1.1 or 1.2), the system predicted the changes faster and get more stable on the track.

I used this idea to improve my model and refine my algorithm. The value of all the mesurements was multiplied/divided by 1.1, increasing the system response (model.py lines 16, 52-56).


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
| Convolution 5x5 | Filter number 24, kernel size 5x5, subsample 2x2 |
| RELU | |
|||
| **3** ||
| Dropout | Rate: 0.3 |
|||
| **4** ||
| Convolution 5x5 | Filter number 36, subsample 2x2 |
| RELU | |
|||
| **5** ||
| Convolution 5x5 | Filter number 48, subsample 2x2 |
| RELU | |
|||
| **6**||
| Dropout | Rate: 0.3 |
|||
| **7** ||
| Convolution 3x3 | Filter number 64
| RELU | |
|||
| **8** ||
| Convolution 3x3 | Filter number 64
| RELU | |
|||
| **9** ||
| Dropout | Rate: 0.3 |
|||
| **10** ||
| Flatten | - |
|||
| **11** ||
| Dense | 100 (Fully Connected Layer) |
|||
| **12** ||
| Dense | 50 (Fully Connected Layer) |
|||
| **13** ||
| Dropout | Rate: 0.3 |
|||
| **14** ||
| Dense | 10 (Fully Connected Layer) |
|||
| **15** ||
| Dense | 1 (Output) |
|||

The model contains dropout layers in order to reduce overfitting, in the end of the model (model.py lines 97, 100, 103, 107). 

#### 3. Training

##### 3.1 Collecting Data & Training Sets

To capture good driving behavior, I first recorded one and two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving - center camera][image2]

The simulation also capture the side images from the track, by different cameras in the car. These images helps to train the network and to stabilize the vehicle. Here a example, of side images (left and right), in the center lane driving.

![Center lane driving - left camera][image3] ![Center lane driving - right camera][image4]

To increase the network performance, I did some extra data collections. The scenarios that I used is described below.

- **1 lap** - 1 lap at the track (on the center lane driving)
- **2 laps** - 2 laps at the track (on the center lane driving)
- **3 laps** - 3 laps at the track (on the center lane driving)
- **2 laps reverse** - 2 laps at the simulation in reverse direction
- **bridge** - cross the bridge simulation
- **reckless** - driving for 2 laps to the board of the road, crossing many times the road without get out of the track
- **reckless reverse** - driving for 1 laps to the board of the road, in the reverse direction, crossing many times the road without get out of the track
- **curve** - drive in a curve after the bridge. In this curve, for many times the car lost its direction.
- **curve reverse** - drive in a curve after the bridge, in reverse direction.

One of the captures that can be highlighted is the bridge set. This test was used to increase the performance of the car on the bridge, where the simulation can lost the direction, because of the different texture of the road. Here some examples of bridge simulation.

![Bridge simulation - center camera][image5] ![Bridge simulation - left camera][image6] ![Bridge simulation - right camera][image7]

##### 3.3 Training Process

To create the model, I combine all the simulation datasets to create a large volume of data samples. I did the following steps to define my best training set. I wanted to use the minimum data set with the best results.

- First, I used a **1 lap training set**. The results were bad;
- Using the **2 laps training set**, the results were so much better;
- Using the **3 laps training set**, the results were as good as using **2 laps training set**, with 30% more cost for training;
- Using the **2 laps training set** with **2 laps reverse**, the results improved a little bit in low speed (9mph) and much more when I increased the speed (18mph);
- Adding **bridge training set** I wanted to test if the results were better in the bridge of the circuit. I think that this sample set is useless, because using the **2 laps training set**, the results were the same. I decided to keep this training set even so, to guarantee the good results;
- Combining **reckless training set** with **2 laps training set**, the results improve a little bit when the car start to get out of the track, turning back quicker;
- Using  **reckless reverse training set**,  **curve training set** and  **curve reverse training set**, the results were worse. I tested these parameters separately and togheter, but the result were always worse.

One other process that I used but had the opposite effect that I expected was sflip the images. Flipping the images, the model become slowly without any great improvemtns. In the end, I decided to not flip the images and only use my training sets.

##### 3.4 Appropriate Training Data

The final training set, defined for my model, was the combination of the following samples:
- **2 laps**
- **2 laps reverse**
- **bridge**
- **reckless training set**

The model was trained for 1 epoch. Train for 2 or 3 epochs make sometimes made few improvements and sometimes made results worse then 1 epoch. The increase of the number of trainings (5, for example) often causes overfitting.

##### 3.5 Training and Validation Data

After the collection process and the definition of the appropriate training data, I had 19074 data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. Then I had 15259 train samples, and 3815 validade samples. 

The neural network with the generator used 5086 images to train the data.

I used this training data for training the model. The validation set helped me to know if the model was over or under fitting. The ideal number of epochs was 1. I used an Adam optimizer so that manually training the learning rate wasn't necessary.


#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

Some improvements also were made to try to increase the performance of the system. The first tuned parameter was the regulation to correct the mesurement for the left and the right images. The parameter was defined as 0.22 (model.py line 11, 63-64), added or subtract in the side images.

The dropout rate of the neural network was defined with 0.3 (model.py line 17, 97, 100, 103, 107). I tested some other parameters, but the best performances that I achieved was when the parameter was defined with 0.3 or 0.4. Changing the neural network for some tests, 0.3 return a more stable results.


