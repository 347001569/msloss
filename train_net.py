import keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import Model
from generator_densenet import My_Custom_Generator
from keras.optimizers import Adam,SGD
from sklearn.utils import shuffle
from loss import ms_loss

def get_model():
    base_model=DenseNet121(include_top=False,weights='imagenet',pooling='avg')
    x=base_model.get_layer(index=-1)
    x=Dense(3975,activation='softmax')(x.output)
    model=Model(base_model.input,x)
    return model

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

    list_label, list_image_path=shuffle(list_label,list_image_path)
    return list_label,list_image_path



def train():
    callback_lists = [
        keras.callbacks.ModelCheckpoint(
            filepath='./desenet_weights/densenet1_weight-{epoch:02d}-{loss:.2f}-{acc:.2f}.h5',
            monitor='loss',
            save_weights_only=True
            ),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=0, mode='auto',
                                          min_delta=0.0001, cooldown=0, min_lr=0)


    ]
    labels,image_paths=get_data()
    model=get_model()
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(My_Custom_Generator(image_paths,labels),epochs=30,callbacks=callback_lists)

train()
