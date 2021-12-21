import cv2
import matplotlib.pyplot as plt
import os


def equalize_this(image_src, with_plot=False, gray_scale=False):

    if not gray_scale:
        r_image, g_image, b_image = cv2.split(image_src)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = cv2.equalizeHist(image_src)
        cmap_val = 'gray'

    return image_eq



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
        image_matrix = cv2.imread(img_name)
        img = equalize_this(image_matrix, with_plot=False, gray_scale=False)

        cv2.imwrite('results/' + img_name.split('\\')[-1], img)

    print('done')

