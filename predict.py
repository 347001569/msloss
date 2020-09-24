"""
Copyright (c) 2019 Intel Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape,normalization
from keras.layers import Lambda
from keras.layers import Conv2D,Convolution2D,TimeDistributed,BatchNormalization,Activation
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D
from  keras.applications.densenet import DenseNet121,preprocess_input
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
# Import base model and siamese model classes from config.py

from generator_pre_emb import My_Custom_Generator
from RoiPoolingConv import RoiPoolingConv


# Import auto roi configuration and library files



# Load tensorflow keras applications path




model_path ='full_model_aug/fullmodel_weight-11-0.00-0.38.h5'
   #



#flag=1 为inference 否则accuracy
flag=0

if flag== 1:
    image_path='dataset/query/id_00000001/02_2_side.jpg'
else:image_path=''
'''
Parse input arguments. Execute the steps in order described below
'''
ap = argparse.ArgumentParser()
# Add flag --gpu x. Default GPU device : 0
ap.add_argument('--gpu', default ='0')
# Add flag --acuracy to calculate total accuracy
ap.add_argument('--accuracy',default= flag-1, action='store_true')
# Add flag --inference to infer an image, add --image with path to image
ap.add_argument('--inference',default=flag,action='store_true')
# Add flag --image with test image path with --inference flag
ap.add_argument('--image', '-i', default=image_path, help = 'Path to image')#
# Add flag --video to do inference on a video file
# Add flag --auto_roi  with --video to perform auto_roi on the input video file
ap.add_argument('--auto_roi', action='store_true')
# Input siamese model
#ap.add_argument('--model', '-m', required = False, help = 'Path to siamese model')
# Input siamese weights
#12
ap.add_argument('--weights', '-w', default=model_path ,required = False, help = 'Path to siamese model weights')


args = ap.parse_args()


# CUDA 9.0
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


'''
Calculate total accuracy across all images
'''
def mapk_eval(features, labels, k):
     results=[]
     # Calculate the distance matrix
     D = euclidean_distances(features, features)
     # Matrix with index values sorted by closes to furthest
     I = np.argsort(D, axis=1)
     # Sort distances by row
     sortedD = np.sort(D)
     # Index of top k predictions
     preds_ys = I[:,:k]
     # Calculate accuracy for all labels
     for i in range(len(labels)):
         mask= sum(np.in1d(labels[i], preds_ys[i])) / float(len(labels[i]))
         results.append(mask)
     return  np.average(results)


'''
Build siamese model
'''
def build_model():
    # Create base model
    base_model=DenseNet121(input_shape=(256,256,3),weights=None,classes=3975)
    out=base_model.output
    x1=base_model.get_layer(index=-3).output

    x2 = base_model.get_layer(name='pool4_conv').output
    x3 =base_model.get_layer(index=-2).output
    print(x3.shape)

    base_model=Model(base_model.input,[out,x1,x2,x3])
    
    # Create feature_model
    roipooling=RoiPoolingConv(8,4)
    input_1 = Input(shape=(16,16,512))
    input_roi=Input(shape=(4,4))
    feature=roipooling([input_1,input_roi])

    feature = TimeDistributed(Convolution2D(64, (1, 1), kernel_initializer='normal'))(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)

    feature = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),padding='same', kernel_initializer='normal'), )(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)

    feature = TimeDistributed(Convolution2D(128, (1, 1), kernel_initializer='normal'), )(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)

    feature_model=Model([input_1,input_roi],feature,name='feature')

    # Create full model

    input_1=Input(shape=(256,256,3),name='input_image')
    input_roi=Input(shape=(4,4),name='input_roi')
    out, x1,x2,x3=base_model(input_1)

    x1=Convolution2D(64, (1, 1), kernel_initializer='normal')(x1)
    x1=BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1=Convolution2D(64, (3, 3),padding='same', kernel_initializer='normal')(x1)
    x1=BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1=Convolution2D(128, (1, 1), kernel_initializer='normal')(x1)
    x1=BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1= Reshape((1,8,8,128),name='reshape_1')(x1)

    feature=feature_model([x2,input_roi])
    cat = Concatenate(axis=1)([x1,feature])
    cat = Flatten(name='flatten_1')(cat)
    cat = Dense(128, use_bias=True, activation='sigmoid')(cat)
    full_model=Model([input_1,input_roi],[out,cat],name='full_model')
    
    full_model.load_weights(args.weights)
    print('===========')

    return base_model, feature_model, full_model

def score_reshape(score, test_preds, query_preds):
    blank = np.zeros((query_preds.shape[0], test_preds.shape[0]), dtype=K.floatx())
    unknown_indices, known_indices = np.indices((query_preds.shape[0],
                                                 test_preds.shape[0]))
    known_indices = known_indices.reshape((known_indices.size,))
    unknown_indices = unknown_indices.reshape((unknown_indices.size,))
    blank[unknown_indices, known_indices] = score.squeeze()
    return blank


def densenet_model():
    base_model = DenseNet121(input_shape=(256, 256, 3), weights=None,
                                 classes=3985)
    base_model.load_weights('desenet_weights/densenet_weights-25-1.00.h5')
    output = base_model.get_layer(index=-2).output


    base_model = Model(base_model.input, output)

    return base_model

'''
Main
'''
if __name__ == '__main__':

    # Load models
    base_model, feature_model, full_model = build_model()
    #base_model=densenet_model()


    #dense=densenet()

    if args.accuracy :

        # NOTE: THIS CODE ASSUMES THE INFERENCE IMAGE CLASSES EXIST IN TEST DATASET

        # Load the test/index images and labels
        #test_imgs = np.load('dataset/npy/anchor_test.npy')           #"./data/data_arrays/anchor_test.npy"
        t_df = pd.read_csv('index/triplet_labels_test.csv')   #".7 /data/data_arrays/triplet_labels_test.csv"
        test_labels = t_df['label']
        test_name = t_df['anchor']
        test_class_id = {}
        num=0
        for idx, i in enumerate(test_labels):

            if str(int(str(i).split('_')[1])) not in test_class_id:
                test_class_id[str(int(str(i).split('_')[1]))] = num
                num+=1

        # Load the query/inference images and labels
        #query_imgs = np.load('dataset/npy/anchor_inference.npy')          #"./data/data_arrays/anchor_inference.npy"
        q_df = pd.read_csv('index/triplet_labels_inference.csv')   #"./data/data_arrays/triplet_labels_inference.csv"
        query_labels = q_df['label']
        query_anchor=q_df['anchor']
        # Create ground truth data for inference images
        y_true = []
        for i in query_labels:
            y_true.append(test_class_id[str(int(str(i).split('_')[1]))])
        y_true = np.asarray(y_true)




        # *********************** PREDICTION USING EMBEDDING MODEL ************************

        # Normalize and get embedding vectors for inference images

        out,query_preds = full_model.predict_generator(My_Custom_Generator(query_anchor.to_list()),
                                                        verbose=1
                                                        )
        # out1, x1, x2,query_preds = base_model.predict_generator(My_Custom_Generator(query_anchor.to_list()),
        #                                                 verbose=1
        #                                                 )

        query_preds = np.asarray(query_preds)

        # Normalize and get embedding vectors for test images


        out2, test_preds = full_model.predict_generator(My_Custom_Generator(test_name.to_list()),
                                                       verbose=1
                                                       )
        # out2, x3, x4, test_preds = base_model.predict_generator(My_Custom_Generator(test_name.to_list()),
        #                                                verbose=1
        #                                                )


        test_preds = np.asarray(test_preds)
        print(test_preds.shape)

        #Normalize the embedding vectors
        query_preds_norm = normalize(query_preds, axis=1)
        test_preds_norm = normalize(test_preds, axis=1)

        # Calculate cosine similarity matrix between the test and inference vectors
        D = cosine_similarity(query_preds_norm, test_preds_norm)


        # Arrange similarity matrix in decreasing order
        I = np.argsort(-D, axis=1)


        # Determine predicted values
        y_pred = []

#TOP K
        k=1
        temp=[]

        for i in range(0,query_preds.shape[0]):
            for j in range(0,k):
                predicted_test_image_file = test_name[I[i][j]]
                x = predicted_test_image_file.split('/')
                predicted_test_label = x[len(x) - 2]
                class_id = str(predicted_test_label).split('_')[1]
            # Append the class of the predicted output file to y_pred
                class_id=str(int(class_id))
                temp.append(test_class_id[class_id])

            if y_true[i] in temp:
                y_pred.append(y_true[i])
            else: y_pred.append(temp[0])

            temp=[]



        # for i in range(0, query_preds.shape[0]):
        #     # Get the predicted image file name
        #     predicted_test_image_file = test_name[I[i][0]]
        #     # Get the class of the predicted output file
        #     x = predicted_test_image_file.split('/')
        #     predicted_test_label = x[len(x) - 2]
        #     class_id = str(predicted_test_label).split('_')[1]
        #     # Append the class of the predicted output file to y_pred
        #     class_id=str(int(class_id))
        #
        #     y_pred.append(test_class_id[class_id])

        # Generate classification report


        print ('Classification report saved to ./reports/report_full_model_cosine_top%s.txt'%k)
        f = open('./reports/report_full_model_cosine_top%s.txt'%k, 'w')
        summary = classification_report(y_true, y_pred)
        f.write(summary)
        f.close()

        # *********************** PREDICTION USING HEAD MODEL ************************



    # Inference on a single image
    if args.inference:
        # Check if test image exists
        if args.image is None:
             assert False, 'Input image to infer using --image'
        if not os.path.exists(args.image):
             assert False, 'Cannot find inference image'
         
        # NOTE: THIS CODE ASSUMES THE INFERENCE IMAGE CLASSES EXIST IN TEST DATASET
        
        # Load the test/index images and labels
        #test_imgs = np.load('dataset/npy/anchor_test.npy')
        t_df = pd.read_csv('index/triplet_labels_test.csv')
        test_labels = t_df['label']
        test_name = t_df['anchor']

        test_class_id = {}
        for idx, i in enumerate(test_labels):
            test_class_id[str(int(str(i).split('_')[1]))] = idx

        # Load the query/inference image 
        query_img = cv2.imread(args.image)
        query_img = cv2.resize(query_img, (256, 256))
        query_img = query_img.reshape(1, 256, 256, 3)
        query_data = np.asarray(query_img)
        query_data = preprocess_input(query_data.astype(np.float32, copy=False))


        # *********************** PREDICTION USING EMBEDDING MODEL ************************

        # Normalize and get embedding vectors for inference images
        out3,y1,y2,query_preds = base_model.predict(query_data)
        query_preds = np.asarray(query_preds)
        np.save('tf_query.npy', query_preds)

        # Normalize and get embedding vectors for test images
        out4,y3,y4,test_preds = base_model.predict_generator(My_Custom_Generator(test_name.to_list()))
        test_preds = np.asarray(test_preds)

        # Normalize the embedding vectors
        query_preds_norm = normalize(query_preds, axis=1)
        test_preds_norm = normalize(test_preds, axis=1)

        # Calculate cosine similarity matrix between the test and inference vectors
        D = cosine_similarity(query_preds_norm, test_preds_norm)
        
        # Arrange similarity matrix in decreasing order
        I = np.argsort(-D, axis=1)

        # Take the top k results
        k = 10
        print('')
        print('Prediction results using embedding model ...')
        print('Top {} results : '.format(k))
        for i in range(0, k):
           print(test_name[I[0][i]])
        

        # *********************** PREDICTION USING HEAD MODEL ************************


    

