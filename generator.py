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
        self.label_choose=3
        self.image_choose=4
        self.image_filenames=image_filenames
        self.labels=labels
        self.batch_size=self.label_choose * self.image_choose



    def __len__(self):
        return (np.ceil(len(self.image_filenames)/float(self.batch_size))).astype(np.int)

    def cv_imread(self, file_path):
        cv_img = cv2.imread(file_path)
        cv_img =cv2.resize(cv_img,(self.img_scale,self.img_scale))
        cv_img =preprocess_input(cv_img)
        return cv_img

    def __getitem__(self, idx):

        label_list=[]
        image_list=[]
        roi_list=[]
        #(x,y,w,h)

        label=[ i for i in range(0, self.num)]
        choose_label=random.sample(label,self.label_choose)
        list_choose_label=list(enumerate(self.labels))
        roi = np.array([[4, 2, 7, 11], [5, 5, 5, 6], [3, 1, 10, 13], [6, 2, 3, 12]])
        for num in choose_label:
            index = [i[0] for i in list_choose_label if i[1] == num]
            choose_index = random.sample(index, self.image_choose)
            for j in choose_index:
                choose_image=self.image_filenames[j]
                image= self.cv_imread(choose_image)

                image_list.append(image)
                label_list.append(num)
                roi_list.append(roi)



        # 预处理
        batch_y_=np_utils.to_categorical(label_list,self.num)


        roi_list=np.array(roi_list)


        return [np.array(image_list),roi_list], [np.array(batch_y_), np.array(label_list)]