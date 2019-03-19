from model import *
from data import *
import numpy as np
import cv2
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def image_normalized(file_path):
    '''
    tif£¬size:512*512£¬gray
    :param dir_path: path to your images directory
    :return:
    '''
    img = cv2.imread(file_path, cv2.COLOR_RGB2GRAY)
    img_shape = img.shape
    image_size = (img_shape[1],img_shape[0])
    img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_standard = cv2.cvtColor(img_standard, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(file_path,img_standard)
    img_new = img_standard
    # img_new = io.imread(file_path, as_gray=True)
    img_new = img_new / 255.
    # img_new = trans.resize(img_new, (256,256), mode='constant')
    img_new = np.reshape(img_new, img_new.shape + (1,)) if (not False) else img_new
    img_new = np.reshape(img_new, (1,) + img_new.shape)
    return img_new,image_size

if __name__ == '__main__':

    #path to images which are prepared to train a model
    train_path = "./data/train"
    image_folder = "image"
    label_folder = "annotation"

    #path to images which aring wating for predicting
    test_path = "./data/test"
    # save the predict images
    save_path = "./data/test"

    dp = data_preprocess(train_path, image_folder, label_folder, test_path, save_path)

    #load model
    model = load_model('./model/crack_model.hdf5')

    # predict image
    dir_path = './data/test'
    for name in os.listdir(dir_path):
        image_path = os.path.join(dir_path,name)
        x,img_size = image_normalized(image_path)
        results = model.predict(x)
        dp.saveResult(results,img_size,name.split('.')[0])