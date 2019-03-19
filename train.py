from model import *
from data import *
import os
import warnings
import keras
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':

    #训练样本集的路径
    train_path = "./data/train"
    image_folder = "image"
    label_folder = "annotation"

    #用于语义分割的图像路径
    test_path = "./data/crack/test_new"

    #保存语义分割结果的路径
    save_path = "./data/crack/test_new"

    #
    dp = data_preprocess(train_path,image_folder,label_folder,test_path,save_path)

    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./log',
    #                                          histogram_freq=1,
    #                                          write_graph=True,
    #                                          write_images=True)

    # 训练模型
    train_data = dp.trainGenerator(batch_size=2)

    model = unet()

    #增量训练的话就用这行替换上面这行即可，增量训练的数据不要和原来的数据放到一起，单独放一个文件夹，
    # 然后将train_path和image_folder，label_folder改为增量数据的文件路径
    # model = load_model('./my_model_0228.hdf5')

    model_checkpoint = keras.callbacks.ModelCheckpoint('./model/crack_model.hdf5', monitor='loss',verbose=1,save_best_only=True)
    #开始训练模型，迭代是10轮，每轮500步
    model.fit_generator(train_data,steps_per_epoch=500,epochs=20,callbacks=[model_checkpoint])

