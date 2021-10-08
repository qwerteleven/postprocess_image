import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import exposure


def hist_RGB_ench(landsat_path):

    # Get RGB bands from Landsat image
    rgb_vector = cv2.imread(landsat_path)



    # Get cutoff values based on standard deviations. Ideally these would be on either side of each histogram peak and cutoff the tail. 
    lims = []
    for i in range(3):
        x = np.mean(rgb_vector[:, i])
        sd = np.std(rgb_vector[:, i])
        low = x-(1.75*sd)  # Adjust the coefficient here if the image doesn't look right
        high = x + (1.75 * sd)  # Adjust the coefficient here if the image doesn't look right
        if low < 0:
            low = 0
        if high > 1:
            high = 1
        lims.append((low, high))

    r_image, g_image, b_image = cv2.split(rgb_vector)

    r = exposure.rescale_intensity(r_image, in_range=lims[0])
    g = exposure.rescale_intensity(g_image, in_range=lims[1])
    b = exposure.rescale_intensity(b_image, in_range=lims[2])
    rgb_enhanced = np.dstack((r, g, b))

    return rgb_enhanced

        

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
    print(len(img_list))
    for img_name in img_list:
        img = hist_RGB_ench(img_name)

        cv2.imwrite('results/' + img_name.split('\\')[-1], img)

    print('done')