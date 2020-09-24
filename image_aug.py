import cv2
import os
form=[cv2.COLORMAP_AUTUMN,cv2.COLORMAP_BONE,cv2.COLORMAP_JET,cv2.COLORMAP_WINTER,cv2.COLORMAP_RAINBOW,cv2.COLORMAP_OCEAN,
      cv2.COLORMAP_SUMMER,cv2.COLORMAP_SPRING,cv2.COLORMAP_COOL,cv2.COLORMAP_HSV,cv2.COLORMAP_PINK,cv2.COLORMAP_HOT]


def get_image_aug(path):
    list_image_aug = []
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for type in form:
        im_color= cv2.applyColorMap(im_gray,type)
        list_image_aug.append(im_color)

    for index,image_aug in enumerate(list_image_aug):
        back_word='_aug%s.jpg'%index
        path_new=path.replace('.jpg',back_word)
        cv2.imwrite(path_new,image_aug)


path_main='dataset/train/'
classes=os.listdir(path_main)
with open('index/triplet_labels_train_aug.csv','w') as f1:
    f1.write('label1,label2,anchor\n')
    for index,num in enumerate(classes):
        images_path=os.listdir(path_main+num)
        for image_name in images_path:
            path=path_main+num+'/'+image_name
            label1=num
            label2=str(index)
            f1.write(label1+','+label2+','+path+'\n')















