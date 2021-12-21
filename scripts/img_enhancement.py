import os
import cv2
import numpy as np
import math
from PIL import Image

def contraste(img, porcentaje):

    #-----Converting image to LAB Color model---
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels---
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel--------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return adjusted


def HDR(img):
    # Read all the files with OpenCV
    files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    images = list([cv2.imread(f) for f in files])
    # Compute the exposure times in seconds
    exposures = np.float32([1. / t for t in [1000, 500, 100, 50, 10]])

    # Compute the response curve
    calibration = cv2.createCalibrateDebevec()
    response = calibration.process(images, exposures)
    # Compute the HDR image
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, exposures, response)

    # Save it to disk
    cv2.imwrite('hdr_image.hdr', hdr)
    durand = cv2.createTonemapDurand(gamma=2.5)
    ldr = durand.process(hdr)

    # Tonemap operators create floating point images with values in the 0..1 range
    # This is why we multiply the image with 255 before saving
    adjusted = cv2.imwrite('durand_image.png', ldr * 255)

    return adjusted

'''

def countTonemap(hdr, min_fraction=0.0005):

	counts, ranges = np.histogram(hdr, 256)
	min_count = min_fraction * hdr.size
	delta_range = ranges[1] - ranges[0]
    
    image = hdr.copy()
    
    for i in range(len(counts)):
        if counts[i] < min_count:
            image[image >= ranges[i + 1]] -= delta_range
        	ranges -= delta_range

    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

'''


def saturacion(img, porcentaje):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = h                    # Hue        [0,179]
    s = s + (s * porcentaje) # Saturation [0,255]
    v = v                    # Value      [0,255].

    final_hsv = cv2.merge((h, s, v))

    adjusted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)   

    return adjusted


def balance_blancos_medio(img):

    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    balance_img = cv2.merge([b, g, r])

    return balance_img



def balance_blancos_dinamico(img):
 
    b, g, r = cv2.split(img)

    """
         Espacio YUV
    """
 
    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
 
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)

    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())

    sum_u = u.sum()
    sum_v = v.sum()
 
    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0

    du = (np.abs(u - avl_u)).sum()
    dv = (np.abs(v - avl_v)).sum()
 
    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0

    radio = 0.5 # Si el valor es demasiado grande o demasiado pequeño, la temperatura del color se desarrollará a dos extremos
 
    u_temp = np.abs(u.astype(np.float32) - (avl_u.astype(np.float32) + avl_du * con_num(avl_u))) < radio * avl_du
    v_temp = np.abs(v.astype(np.float32) - (avl_v.astype(np.float32) + avl_dv * con_num(avl_v))) < radio * avl_dv
    temp = u_temp | v_temp
    num_y[temp] = y[temp]

    for i in range(m):
        for j in range(n):
            if temp[i][j] > 0:
                yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
    ysum = (temp).sum()
 
    sum_yhistogram = 0


    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num> 0.1 * ysum: # Toma el primer 10% de los puntos brillantes como valor calculado.
                            # Si el valor es demasiado grande, es fácil sobreexponer. Si el valor es demasiado pequeño,
                            # el rango de ajuste es pequeño
            key = Y
            break
        Y = Y - 1

    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
 

 
    num_rgb = (num_y > key).sum()
    sum_r = r[num_y > key].sum()
    sum_g = g[num_y > key].sum()
    sum_b = b[num_y > key].sum()
 
    if num_rgb == 0:
        return img
    
    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb
 
    b_point = b.astype(np.float32) * int(max_y) / avl_b
    g_point = g.astype(np.float32) * int(max_y) / avl_g
    r_point = r.astype(np.float32) * int(max_y) / avl_r
 
    b_point[b_point > 255] = 255
    b_point[b_point < 0] = 0
    b = b_point.astype(np.uint8)
 
    g_point[g_point > 255] = 255
    g_point[g_point < 0] = 0
    g = g_point.astype(np.uint8)
 
    r_point[r_point > 255] = 255
    r_point[r_point < 0] = 0
    r = r_point.astype(np.uint8)
 
    return cv2.merge([b, g, r])



def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def convert_temp(image, porcentaje):

    # normal value for photo 5000K

    r, g, b = kelvin_table[5000 + roundup((porcentaje * 100))]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0)


    return image.convert('RGB', matrix)


def convert_temp_by_lamp(image, ligth):
    r, g, b = light_table[ligth]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0)
    return image.convert('RGB', matrix)
    

light_table = {

	'Candle':                     (255, 147, 41),
	'40W Tungsten':               (255, 197, 143),
	'100W Tungsten' :             (255, 214, 170),
	'Halogen' :                   (255, 241, 224),
	'Carbon Arc' :                (255, 250, 244),
	'High Noon Sun':              (255, 255, 251),
	'Direct Sunlight' :           (255, 255, 255),
	'Overcast Sky' :              (201, 226, 255),
	'Clear Blue Sky' :            (64, 156, 255),
	'Warm Fluorescent':           (255, 244, 229),
	'Standard Fluorescent' :      (244, 255, 250),
	'Cool White Fluorescent' :    (212, 235, 255),
	'Full Spectrum Fluorescent' : (255, 244, 242),
	'Grow Light Fluorescent':     (255, 239, 247),
	'Black Light Fluorescent' :   (167, 0, 255),
	'Mercury Vapor' :             (216, 247, 255),
	'Sodium Vapor':               (255, 209, 178),
	'Metal Halide':               (242, 252, 255),
	'High Pressure Sodium' :      (255, 183, 76)
}


kelvin_table = {
    1000: (255, 56, 0),
    1100: (255, 71, 0),
    1200: (255, 83, 0),
    1300: (255, 93, 0),
    1400: (255, 101, 0),
    1500: (255, 109, 0),
    1600: (255, 115, 0),
    1700: (255, 121, 0),
    1800: (255, 126, 0),
    1900: (255, 131, 0),
    2000: (255, 138, 18),
    2100: (255, 142, 33),
    2200: (255, 147, 44),
    2300: (255, 152, 54),
    2400: (255, 157, 63),
    2500: (255, 161, 72),
    2600: (255, 165, 79),
    2700: (255, 169, 87),
    2800: (255, 173, 94),
    2900: (255, 177, 101),
    3000: (255, 180, 107),
    3100: (255, 184, 114),
    3200: (255, 187, 120),
    3300: (255, 190, 126),
    3400: (255, 193, 132),
    3500: (255, 196, 137),
    3600: (255, 199, 143),
    3700: (255, 201, 148),
    3800: (255, 204, 153),
    3900: (255, 206, 159),
    4000: (255, 209, 163),
    4100: (255, 211, 168),
    4200: (255, 213, 173),
    4300: (255, 215, 177),
    4400: (255, 217, 182),
    4500: (255, 219, 186),
    4600: (255, 221, 190),
    4700: (255, 223, 194),
    4800: (255, 225, 198),
    4900: (255, 227, 202),
    5000: (255, 228, 206),
    5100: (255, 230, 210),
    5200: (255, 232, 213),
    5300: (255, 233, 217),
    5400: (255, 235, 220),
    5500: (255, 236, 224),
    5600: (255, 238, 227),
    5700: (255, 239, 230),
    5800: (255, 240, 233),
    5900: (255, 242, 236),
    6000: (255, 243, 239),
    6100: (255, 244, 242),
    6200: (255, 245, 245),
    6300: (255, 246, 247),
    6400: (255, 248, 251),
    6500: (255, 249, 253),
    6600: (254, 249, 255),
    6700: (252, 247, 255),
    6800: (249, 246, 255),
    6900: (247, 245, 255),
    7000: (245, 243, 255),
    7100: (243, 242, 255),
    7200: (240, 241, 255),
    7300: (239, 240, 255),
    7400: (237, 239, 255),
    7500: (235, 238, 255),
    7600: (233, 237, 255),
    7700: (231, 236, 255),
    7800: (230, 235, 255),
    7900: (228, 234, 255),
    8000: (227, 233, 255),
    8100: (225, 232, 255),
    8200: (224, 231, 255),
    8300: (222, 230, 255),
    8400: (221, 230, 255),
    8500: (220, 229, 255),
    8600: (218, 229, 255),
    8700: (217, 227, 255),
    8800: (216, 227, 255),
    8900: (215, 226, 255),
    9000: (214, 225, 255),
    9100: (212, 225, 255),
    9200: (211, 224, 255),
    9300: (210, 223, 255),
    9400: (209, 223, 255),
    9500: (208, 222, 255),
    9600: (207, 221, 255),
    9700: (207, 221, 255),
    9800: (206, 220, 255),
    9900: (205, 220, 255),
    10000: (207, 218, 255),
    10100: (207, 218, 255),
    10200: (206, 217, 255),
    10300: (205, 217, 255),
    10400: (204, 216, 255),
    10500: (204, 216, 255),
    10600: (203, 215, 255),
    10700: (202, 215, 255),
    10800: (202, 214, 255),
    10900: (201, 214, 255),
    11000: (200, 213, 255),
    11100: (200, 213, 255),
    11200: (199, 212, 255),
    11300: (198, 212, 255),
    11400: (198, 212, 255),
    11500: (197, 211, 255),
    11600: (197, 211, 255),
    11700: (197, 210, 255),
    11800: (196, 210, 255),
    11900: (195, 210, 255),
    12000: (195, 209, 255)}


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



def main():

    img_list = get_all_names()

    for img_name in img_list:
        
        img = Image.open(img_name)
        img = img.convert('RGB')
        
        img = contraste(img, 60)
        # img = saturacion(img, 40)
        img = convert_temp(img, 15)
        img = balance_blancos_medio(img)

        cv2.imwrite('results/' + img_name.split('\\')[-1], img)

    print('done')


if __name__ == '__main__':
    main()






