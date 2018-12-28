from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings
warnings.filterwarnings("ignore")


Sky = [255,255,255]

other = [0,0,0]
COLOR_DICT = np.array([Sky,other])

class data_preprocess:
    def __init__(self,train_path,image_folder,label_folder,test_path,save_path,img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.train_path = train_path
        self.image_folder= image_folder
        self.label_folder = label_folder
        self.test_path = test_path
        self.save_path = save_path
        self.data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
        self.image_color_mode = "grayscale"
        self.label_color_mode = "grayscale"

        self.flag_multi_class = False
        self.num_class = 2
        self.target_size = (256, 256)

        self.img_type = 'tif'

    def adjustData(self,img,label,flag_multi_class,num_class):
        if(flag_multi_class):
            img = img / 255.
            label = label[:,:,:,0] if(len(label.shape) == 4) else label[:,:,0]
            new_label = np.zeros(label.shape + (num_class,))
            for i in range(num_class):
                new_label[label == i,i] = 1
            new_label = np.reshape(new_label,(new_label.shape[0],new_label.shape[1]*new_label.shape[2],new_label.shape[3])) if flag_multi_class else np.reshape(new_label,(new_label.shape[0]*new_label.shape[1],new_label.shape[2]))
            label = new_label
        elif(np.max(img) > 1):
            img = img / 255.
            label = label /255.
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return (img,label)

    def image_normalized(self,dir_path):
        for file_name in os.listdir(dir_path):
            if os.path.splitext(file_name)[1].replace('.','') == self.img_type:
                jpg_name = os.path.join(dir_path, file_name)
                img = cv2.imread(jpg_name, cv2.COLOR_RGB2GRAY)
                img_new = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(jpg_name, img_new)


    def trainGenerator(self,batch_size,image_save_prefix = "image",label_save_prefix = "label",
                       save_to_dir = None,seed = 7):
        '''
        can generate image and label at the same time
        use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''
        # self.image_normalized(os.path.join(self.train_path,self.image_folder))
        # self.image_normalized(os.path.join(self.train_path,self.label_folder))
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes = [self.image_folder],
            class_mode = None,
            color_mode = self.image_color_mode,
            target_size = self.target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = image_save_prefix,
            seed = seed)
        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes = [self.label_folder],
            class_mode = None,
            color_mode = self.label_color_mode,
            target_size = self.target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = label_save_prefix,
            seed = seed)
        train_generator = zip(image_generator, label_generator)
        for (img,label) in train_generator:
            img,label = self.adjustData(img,label,self.flag_multi_class,self.num_class)
            yield (img,label)


    def testGenerator(self):
        # self.image_normalized(self.test_path)
        filenames = os.listdir(self.test_path)
        for filename in filenames:
            img = io.imread(os.path.join(self.test_path,filename),as_gray=True)
            img = img / 255.
            img = trans.resize(img,self.target_size,mode='constant')
            img = np.reshape(img,img.shape+(1,)) if (not self.flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)
            yield img


    def geneTrainNpy(self,image_path,label_path,image_prefix = "image",label_prefix = "label",image_as_gray = True,label_as_gray = True):
        image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
        image_arr = []
        label_arr = []
        for index,item in enumerate(image_name_arr):
            img = io.imread(item,as_gray = image_as_gray)
            img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
            label = io.imread(item.replace(image_path,label_path).replace(image_prefix,label_prefix),as_gray = label_as_gray)
            label = np.reshape(label,label.shape + (1,)) if label_as_gray else label
            img,label = self.adjustData(img,label,self.flag_multi_class,self.num_class)
            image_arr.append(img)
            label_arr.append(label)
        image_arr = np.array(image_arr)
        label_arr = np.array(label_arr)
        return image_arr,label_arr


    def labelVisualize(self,color_dict,img):
        img = img[:,:,0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(self.num_class):
            img_out[img == i,:] = color_dict[i]
        return img_out / 255.



    def saveResult(self,npyfile):
        for i,item in enumerate(npyfile):
            img = self.labelVisualize(self.num_class,COLOR_DICT,item) if self.flag_multi_class else item[:,:,0]
            cv2.imwrite(os.path.join(self.save_path,("%d_predict."+self.img_type)%i),img)