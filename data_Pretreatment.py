#encoding:utf-8
import  os
import cv2

'''
before your train or predict you should transfrom your images to standard format
'''

def image_normalized(dir_path,save_dir):
    '''
    tif£¬size:512*512£¬gray
    :param dir_path: path to your images directory
    :param save_dir: path to your images after normalized
    :return:
    '''
    for file_name in os.listdir(dir_path):
        if os.path.splitext(file_name)[1].replace('.', '') == "tif":
            jpg_name = os.path.join(dir_path, file_name)
            save_path = os.path.join(save_dir,file_name)
            img = cv2.imread(jpg_name, cv2.COLOR_RGB2GRAY)
            img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_standard = cv2.cvtColor(img_standard, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path, img_standard)

if __name__ == '__main__':
    image_normalized('./data/image','./data/image_new')
