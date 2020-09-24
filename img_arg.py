import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
path_main='dataset/train/'
classes=os.listdir(path_main)

seq = iaa.Sequential([

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    # iaa.Sometimes(
    #     0.5,
    #     iaa.GaussianBlur(sigma=(0, 0.5))
    # ),

    # iaa.WithColorspace('BGR',
    #                from_colorspace='RGB',
    #                children=None,
    #                name=None,
    #                deterministic=False,
    #                random_state=None),

    # Strengthen or weaken the contrast in each image.
    # iaa.LinearContrast((0.75, 1.5)),
    # iaa.Dropout2d(p=0.8),
    iaa.Add((-40, 40), per_channel=0.5),

    # iaa.Dropout(p=0.1),

    # iaa.ContrastNormalization(alpha=1, per_channel=False, name=None, deterministic=False, random_state=None),
    # iaa.Invert(p=0.1, per_channel=False, min_value=0, max_value=255, name=None, deterministic=False, random_state=None),
    # iaa.MultiplyElementwise(mul=1, per_channel=False, name=None, deterministic=False, random_state=None),
    # iaa.ChangeColorspace('HSV', from_colorspace='RGB', alpha=1.0, name=None, deterministic=False, random_state=None),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Sharpen(alpha=0, lightness=1, name=None, deterministic=False, random_state=None),

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True) # apply augmenters in random order

for num in classes:
    images_path=os.listdir(path_main+num)
    for image_name in images_path:
        path=path_main+num+'/'+image_name
        path_new=path.replace('.jpg','_aug1.jpg')
        image=cv2.imread(path)
        image=np.reshape(image,(1,256,256,3))

        images_aug = seq(images=image)[0]
        #cv2.imwrite(path_new,images_aug)








