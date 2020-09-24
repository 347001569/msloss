import os
import json
path_main='train/'
list_image_type=os.listdir(path_main)
with open('index_image_name.csv','w',encoding='utf-8') as f1, open('labels.json','r') as f2:
    dict_image=json.load(f2)
    f1.write('label,image_path\n')

    for image_type in list_image_type:
        list_image_name=os.listdir(path_main+image_type)
        for image_name in list_image_name:
            image_path=path_main+image_type+'/'+image_name
            label=str(dict_image[image_type])
            f1.write(label+','+image_path+'\n')



            print(label,image_path)