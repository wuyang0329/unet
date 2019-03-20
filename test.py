#encoding:utf-8
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
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    img = cv2.imread(file_path)
    img_shape = img.shape
    image_size = (img_shape[1],img_shape[0])
    img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_new = img_standard
    img_new = np.asarray([img_new / 255.])
    return img_new,image_size


if __name__ == '__main__':

    #path to images which are prepared to train a model
    train_path = "./data/crack/train"
    image_folder = "image"
    label_folder = "annotation"

    #path to images which aring wating for predicting
    test_path = "./data/road/test/images"

    # save the predict images
    save_path = "./data/road/test/predict"

    dp = data_preprocess(train_path, image_folder, label_folder, test_path, save_path)

    #load model
    model = load_model('./model/road_model_v3.hdf5')

    for name in os.listdir(test_path):
        image_path = os.path.join(test_path,name)
        x,img_size = image_normalized(image_path)
        results = model.predict(x)
        img_standard = cv2.cvtColor(results[0], cv2.COLOR_GRAY2RGB)
        dp.saveResult(np.asarray([results[0]]),img_size,name.split('.')[0])
