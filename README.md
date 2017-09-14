## Behavioral Cloning

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) summarizing the results
* [video.mp4](./video.mp4) containing the output video file

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 
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
