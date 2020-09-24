#写一个batch迭代器
import keras
import numpy as np
import cv2
from keras.applications.densenet import preprocess_input
from keras.utils import np_utils
import random
class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self,image_filenames,labels):
        self.img_scale=256
        self.num=3975
        self.image_filenames=image_filenames
        self.labels=labels
        self.batch_size=16



    def __len__(self):
        return (np.ceil(len(self.image_filenames)/float(self.batch_size))).astype(np.int)

    def cv_imread(self, file_path):
        cv_img = cv2.imread(file_path)
        cv_img =cv2.resize(cv_img,(self.img_scale,self.img_scale))
        cv_img =preprocess_input(cv_img)
        return cv_img

    def __getitem__(self, idx):
        batch_x=self.image_filenames[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y=self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        image_list=[]
        for path in batch_x:
            image=self.cv_imread(path)
            image_list.append(image)

        image_list=np.array(image_list).reshape((-1,self.img_scale,self.img_scale,3))
        batch_y_=np_utils.to_categorical(batch_y,self.num)




        return image_list, batch_y_