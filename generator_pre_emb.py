from keras.applications.densenet import preprocess_input
from keras.utils import np_utils
import keras
import numpy as np
import cv2
class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self,image_filenames,batch_size=128):
        super(My_Custom_Generator, self).__init__()
        self.image_filenames=image_filenames
        self.batch_size=batch_size


    def __len__(self):
        return (np.ceil(len(self.image_filenames)/float(self.batch_size))).astype(np.int)



    def __getitem__(self, idx):
        batch_x=self.image_filenames[idx*self.batch_size : (idx+1)*self.batch_size]
        roi = np.array([[4, 2, 7, 11], [5, 5, 5, 6], [3, 1, 10, 13], [6, 2, 3, 12]])

        # 预处理

        list_image=[]
        roi_list = []
        for file_name in batch_x:
            file_name=file_name.replace('\n','')
            image=cv2.imread(str(file_name))
            #image=cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            #image=image.reshape(256,256,3)
            image=preprocess_input(image.astype(np.float32, copy=False))
            list_image.append(image)
            roi_list.append(roi)


        data_x=np.array([image_batch for image_batch in list_image])

        return data_x, np.array(a for a in roi_list )