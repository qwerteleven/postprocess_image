import os
from PIL import Image


def get_path_image():
    
    img_list = []

    thisdir = os.getcwd() + '/images'
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith(".jpeg"):
                instance = os.path.join(r, file)
                img_list.append(instance)

    return img_list

def get_all_names():
    img_list = []

    thisdir = os.getcwd() + '/images'
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith(".jpeg"):
                instance = os.path.join(r, file)
                img_list.append(instance)

    file = open('names.txt', 'a')

    for img_name in img_list:
        file.write(img_name.split('\\')[-1].split('.')[0] + '\n')

    file.close()



if __name__ == '__main__':
    get_all_names()
    print('done')







