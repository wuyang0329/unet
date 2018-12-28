### unet
Keras implementation of unet for Gray pictures.
my original image's type is png ,and i convert its type to tif.
### Data
You can download train && test data from here:http://www.cvlibs.net/download.php?file=data_road.zip

### How to use
## Requirement
- OpenCV
- Python 3.6
- Tensorflow-gpu-1.8.0
- Keras-2.2.4
## train and test 
run the main.py to train you own model and predict the test images
you should give your path to the train and test images 

### Results
The model is trained for 10 epochs.

After 10 epochs, calculated accuracy is about 0.97, the loss is about 0.07

Loss function for the training is basically just a binary crossentropy.

![img/0test.jpg](img/0test.jpg)

![img/0label.jpg](img/0label.jpg)


## About
Unet is More commonly used in medical areas.

## Reference
https://github.com/zhixuhao/unet


