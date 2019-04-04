### unet
Keras implementation of unet.
### Data
You can download:

Kitti dataset from here:http://www.cvlibs.net/download.php?file=data_road.zip

CamVid dataset from here:https://github.com/preddy5/segnet/tree/master/CamVid

### How to use
## Requirement
- OpenCV
- Python 3.6
- Tensorflow-gpu-1.8.0
- Keras-2.2.4
## train and test
Before you start training, you must make sure your dataset have the right format

If you just two classes to  classify, you should set flag_multi_class equal to False and num_class=2

if you have many classes to classify, you should set flag_multi_class equal to True and num_class=number of your classes

Then you should set image type , image_color_mode and label_color_mode.

change the data path and run the train.py to train you own model and test.py to predict the test images

you should give your path to the train and test images 

### Results
The binary classify model is trained for 30 epochs.
After 30 epochs, calculated accuracy is about 0.989, the loss is about 0.02
Loss function for the training is basically just a binary crossentropy.
![image/test.png](image/test.png)
![image/test_predict.png](image/test_predict.png)


## About
Unet is More commonly used in medical areas.

## Reference
https://github.com/zhixuhao/unet


