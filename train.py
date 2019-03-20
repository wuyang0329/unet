#encoding:utf-8
# from model import *
from model_rgb import  *
from data import *
import numpy as np
import cv2
import os
import skimage.io as io
import skimage.transform as trans
import warnings
import keras
from keras import Model
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':

    #path to images which are prepared to train a model
    train_path = "./data/road/train"
    image_folder = "images_new"
    label_folder = "labels_new"

    #path to images which you want to seg
    test_path = "./road/test/images"

    #save the predict images
    save_path = "./road/test/predict"

    dp = data_preprocess(train_path,image_folder,label_folder,test_path,save_path)

    # train your own model
    train_data = dp.trainGenerator(batch_size=2)
    model = unet()

    model_checkpoint = keras.callbacks.ModelCheckpoint('./model/road_model_v3.hdf5', monitor='loss',verbose=1,save_best_only=True)
    model.fit_generator(train_data,steps_per_epoch=300,epochs=20,callbacks=[model_checkpoint])

