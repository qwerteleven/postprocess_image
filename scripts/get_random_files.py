import os
import random
from shutil import copyfile

def get_all_names():
    img_list = []

    thisdir = 'images'
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith(".jpeg"):
                instance = os.path.join(r, file)
                img_list.append(instance)

    return img_list
    

 
if __name__ == '__main__':


    img_list = get_all_names()

    selection = [img.split('/')[-1] for img in random.choices(img_list, k = 100)]


    for img in selection:
        copyfile('images/' + img, 'selection/GT/' + img)
        copyfile('results/dynamic_threshold_white_balance/' + img, 'selection/dynamic_threshold_white_balance/' + img)
    
    

