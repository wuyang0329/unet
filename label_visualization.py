import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) > len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation2(image, seg_map):
    """
    输入图片和分割 mask 的统一可视化.
    """
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.figure()
    plt.imshow(seg_image)
    plt.imshow(image,alpha=0.5)
    plt.axis('off')
    plt.show()


test_path = "CamVid\\test"
predict_path =  "CamVid\\predict"
for filename in os.listdir(test_path):
    imgfile = os.path.join(test_path,filename)
    pngfile = os.path.join(predict_path,filename.split('.')[0]+"_predict.png")
    img = cv2.imread(imgfile, 1)
    img = img[:,:,::-1]
    seg_map = cv2.imread(pngfile, 0)
    vis_segmentation2(img, seg_map)
