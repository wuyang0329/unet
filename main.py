from model import *
from data import *
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    train_path = "./data/road/train"
    image_folder = "image"
    label_folder = "label"
    test_path = "./data/road/test"
    save_path = "./data/road/test"
    dp = data_preprocess(train_path,image_folder,label_folder,test_path,save_path)

    # train_data = dp.trainGenerator(batch_size=2,save_to_dir='./data/road/aug')
    #
    # model = unet()
    # model_checkpoint = ModelCheckpoint('my_model_1227_2135.hdf5', monitor='loss',verbose=1, save_best_only=True)
    # model.fit_generator(train_data,steps_per_epoch=500,epochs=1,callbacks=[model_checkpoint])

    model = load_model('./my_model_1227_2135.hdf5')
    test_data = dp.testGenerator()
    results = model.predict_generator(test_data,steps=30,verbose=1)
    dp.saveResult(results)