#encoding:utf-8
from model_v2 import  *
from data import *
import os
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':

    #path to images which are prepared to train a model
    train_path = "data/CamVid"
    image_folder = "train"
    label_folder = "trainannot"
    log_filepath = './log'
    flag_multi_class = True
    num_class = 12
    dp = data_preprocess(train_path=train_path,
                         image_folder=image_folder,label_folder=label_folder, 
                         flag_multi_class = flag_multi_class,
                         num_classes=num_class)

    # train your own model
    train_data = dp.trainGenerator(batch_size=2)
    test_data = dp.testGenerator()
    model = unet(num_class=12)

    tb_cb = TensorBoard(log_dir=log_filepath)
    model_checkpoint = keras.callbacks.ModelCheckpoint('./model/road_model.hdf5', monitor='loss',verbose=1,save_best_only=True)
    history = model.fit_generator(train_data,steps_per_epoch=200,epochs=30,callbacks=[model_checkpoint,tb_cb])

    # draw the loss and accuracy curve
    plt.figure(12, figsize=(6, 6), dpi=60)
    plt.subplot(211)
    plt.plot(history.history['loss'], label='train')
    plt.title('loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(history.history['acc'], label='train')
    plt.title('acc')
    plt.legend()

    plt.show()
