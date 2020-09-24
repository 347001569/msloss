import keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense,GlobalAvgPool2D,Convolution2D,BatchNormalization,Activation,Reshape
from keras.models import Model
from generator import My_Custom_Generator
from RoiPoolingConv import RoiPoolingConv
from keras.layers import Concatenate,TimeDistributed,Input,Flatten
from keras.optimizers import SGD,Adam
from FixedBatchNormalization import FixedBatchNormalization
from loss import ms_loss
import math


batch_size=12
classes=3975



def get_base_model():
    base_model=DenseNet121(input_shape=(256,256,3),weights='desenet_weights/densenet1_weight-03-0.01-1.00.h5',classes=3975)
    out=base_model.output

    x1=base_model.get_layer(index=-3).output
    x2 = base_model.get_layer(name='pool4_conv').output
    base_model=Model(base_model.input,[out,x1,x2])
    return base_model

def get_feature_model():
    embedding_model = get_base_model()
    roipooling=RoiPoolingConv(8,4)
    print(embedding_model.output_shape)
    input_1 = Input(batch_shape=(batch_size,16,16,512))
    input_roi=Input(batch_shape=(batch_size,4,4))
    feature=roipooling([input_1,input_roi])
    print(feature.shape)

    feature = TimeDistributed(Convolution2D(64, (1, 1), kernel_initializer='normal'))(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    #feature = TimeDistributed(FixedBatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)

    feature = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),padding='same', kernel_initializer='normal'), )(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    #feature = TimeDistributed(FixedBatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)
    print(feature.shape)

    feature = TimeDistributed(Convolution2D(128, (1, 1), kernel_initializer='normal'), )(feature)
    feature = TimeDistributed(BatchNormalization(axis=3))(feature)
    #feature = TimeDistributed(FixedBatchNormalization(axis=3))(feature)
    feature = Activation('relu')(feature)


    feature_model=Model([input_1,input_roi],feature,name='feature')
    return feature_model

def get_full_model():
    base_model=get_base_model()
    feature_model=get_feature_model()
    input_1=Input(batch_shape=(batch_size,256,256,3),name='input_image')
    input_roi=Input(batch_shape=(batch_size,4,4),name='input_roi')
    out, x1,x2=base_model(input_1)

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
    cat = Dense(512, use_bias=True, activation='sigmoid')(cat)
    full_model=Model([input_1,input_roi],[out,cat],name='full_model')

    return full_model



def get_data():
    list_label=[]
    list_image_path=[]
    with open('index/triplet_labels_train_aug.csv','r') as f1:
        next(f1)
        reader=f1.readlines()
        for read in reader:
            read=read.replace('\n','').split(',')
            label=int(read[1])
            image_path=read[2]
            list_label.append(label)
            list_image_path.append(image_path)


    return list_label,list_image_path

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

def train():
    flag=True

    callback_lists=[
        # keras.callbacks.EarlyStopping(monitor='acc',
        #                               patience=1,
        #                               ),
        keras.callbacks.ModelCheckpoint(filepath= './full_model_512/fullmodel_weight-{epoch:02d}-{model_1_loss:.2f}-{dense_1_loss:.2f}.h5',
                                        monitor='loss',
                                        save_weights_only=True
                                        ),
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                   factor=factor,
        #                                   patience=patience,
        #                                   min_lr=min_lr),

        # keras.callbacks.LambdaCallback(
        # on_epoch_begin=lambda epoch: json_log.write(
        #     json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        # ),

        keras.callbacks.LearningRateScheduler(schedule=step_decay,
                                              verbose=1)


    ]
    labels,image_paths=get_data()
    model=get_full_model()
    model.summary()

    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),loss=['categorical_crossentropy',ms_loss(ms_mining=flag)],metrics=['accuracy'],loss_weights=[1,3],)
    model.fit_generator(My_Custom_Generator(image_paths,labels),epochs=1,callbacks=callback_lists)


train()