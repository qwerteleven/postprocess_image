import cv2
import numpy as np
import math
from PIL import Image,ImageEnhance
import subprocess
import colorsys
import os


def shift_hue(arr, hout):

    r, g, b, a = np.rollaxis(arr, axis=-1)
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img


def enhancement_CLAHE(img_name):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img_name, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



def adjust_gamma(img, gamma):

    """
    
    Build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.

    """

    # code from
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

 
def perfect_reflective_white_balance(img_input):

    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    sum_ = b.astype(np.int32) + g.astype(np.int32) + r.astype(np.int32)
 
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    sum_b = b[sum_ >= key].sum()
    sum_g = g[sum_ >= key].sum()
    sum_r = r[sum_ >= key].sum()
    time = (sum_ >= key).sum()
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
 
    b = img[:, :, 0].astype(np.int32) * maxvalue / int(avg_b)
    g = img[:, :, 1].astype(np.int32) * maxvalue / int(avg_g)
    r = img[:, :, 2].astype(np.int32) * maxvalue / int(avg_r)
    b[b > 255] = 255
    b[b < 0] = 0
    g[g > 255] = 255
    g[g < 0] = 0
    r[r > 255] = 255
    r[r < 0] = 0
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
 
    return img
 
 
def gray_world_assumes_white_balance(img):

  
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
 
    Ba[Ba > 255] = 255
    Ga[Ga > 255] = 255
    Ra[Ra > 255] = 255
 
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra

    return dst_img
 
 
def color_correction_of_image_analysis(img):
 
    b, g, r = cv2.split(img)
    m, n = b.shape
 
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
 
    I_r_2 = (r.astype(np.float32) ** 2).astype(np.float32)
    I_b_2 = (b.astype(np.float32) ** 2).astype(np.float32)

    sum_I_r_2 = I_r_2.sum()
    sum_I_b_2 = I_b_2.sum()
    sum_I_g = g.sum()
    sum_I_r = r.sum()
    sum_I_b = b.sum()
 
    max_I_r = r.max()
    max_I_g = g.max()
    max_I_b = b.max()
    max_I_r_2 = I_r_2.max()
    max_I_b_2 = I_b_2.max()
 
    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])

    b_point = u_b * (b.astype(np.float32) ** 2) + v_b * b.astype(np.float32)
    r_point = u_r * (r.astype(np.float32) ** 2) + v_r * r.astype(np.float32)
 
    b_point[b_point > 255] = 255
    b_point[b_point < 0] = 0
    b = b_point.astype(np.uint8)
 
    r_point[r_point > 255] = 255
    r_point[r_point < 0] = 0
    r = r_point.astype(np.uint8)
 
    return cv2.merge([b, g, r])
 

 
def dynamic_threshold_white_balance(img):

 
    b, g, r = cv2.split(img)
 
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
    radio = 0.5 
 
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
        if num> 0.1 * ysum: 
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
    # print("num_rgb", num_rgb)
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
 
 
def gamma_trans(img):

 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma = math.log10 (0.5) / math.log10 (mean / 255) 
    gamma_table = [np.power (x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round (np.array (gamma_table)). astype (np.uint8)
    return cv2.LUT (img, gamma_table)
 
 
def contrast_image_correction(img):

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    mv = cv2.split(img_yuv)
    img_y = mv[0].copy()
 
    temp = cv2.bilateralFilter(mv[0], 9, 50, 50)

    exp = np.power(2, (128.0 - (255 - temp).astype(np.float32)) / 128.0)
    temp = (255 * np.power(img_y.flatten() / 255.0, exp.flatten())).astype(np.uint8)
    temp = temp.reshape((img_y.shape))
 
    dst = img.copy()
 
    img_y[img_y == 0] = 1
    for k in range(3):
        val = temp / img_y
        val1 = img[:, :, k].astype(np.int32) + img_y.astype(np.int32)
        val2 = (val * val1 + img[:, :, k] - img_y) / 2
        dst[:, :, k] = val2.astype(np.int32)
 
    return dst


def WhiteBlance(img, mode=1):

    """

    White balance processing 

    1 mean
    2 perfect reflection
    3 grayscale world
    4 based image analysis and color correction
    5 dynamic threshold

    """

    #  Read image
    b, g, r = cv2.split(img)
    #  Mean is three-channel
    h, w, c = img.shape
    if mode == 2:
        #  Perfect reflection white balance --- Relying on the Ratio value to choose and the largest area of ​​the brightness is not a white image effect.
        output_img = img.copy()
        sum_ = np.double() + b + g + r
        hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
        Y = 765
        num, key = 0, 0
        ratio = 0.01
        while Y >= 0:
            num += hists[Y]
            if num > h * w * ratio / 100:
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(sum_ >= key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        times = len(sumkey[0])
        avg_b, avg_g, avg_r = sum_b / times, sum_g / times, sum_r / times

        maxvalue = float(np.max(output_img))
        output_img[:, :, 0] = output_img[:, :, 0] * maxvalue / int(avg_b)
        output_img[:, :, 1] = output_img[:, :, 1] * maxvalue / int(avg_g)
        output_img[:, :, 2] = output_img[:, :, 2] * maxvalue / int(avg_r)
    elif mode == 3:
        #  Grayscale world hypothesis
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        #  Need to adjust the gain of RGB components
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        ba, ga, ra = b * kb, g * kg, r * kr

        output_img = cv2.merge([ba, ga, ra])
    elif mode == 4:
        #  Image analysis - based bias detection and color correction
        I_b_2, I_r_2 = np.double(b) ** 2, np.double(r) ** 2
        sum_I_b_2, sum_I_r_2 = np.sum(I_b_2), np.sum(I_r_2)
        sum_I_b, sum_I_g, sum_I_r = np.sum(b), np.sum(g), np.sum(r)
        max_I_b, max_I_g, max_I_r = np.max(b), np.max(g), np.max(r)
        max_I_b_2, max_I_r_2 = np.max(I_b_2), np.max(I_r_2)
        [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
        [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
        b0 = np.uint8(u_b * (np.double(b) ** 2) + v_b * b)
        r0 = np.uint8(u_r * (np.double(r) ** 2) + v_r * r)
        output_img = cv2.merge([b0, g, r0])
    elif mode == 5:
        #  Dynamic threshold algorithm ---- white point detection and white point adjustment
        #  Only white point detection is not the same as white point as the perfect reflection algorithm, but is determined by another rule.
        def con_num(x):
            if x > 0:
                return 1
            if x < 0:
                return -1
            if x == 0:
                return 0

        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #  YUV space
        (y, u, v) = cv2.split(yuv_img)
        max_y = np.max(y.flatten())
        sum_u, sum_v = np.sum(u), np.sum(v)
        avl_u, avl_v = sum_u / (h * w), sum_v / (h * w)
        du, dv = np.sum(np.abs(u - avl_u)), np.sum(np.abs(v - avl_v))
        avl_du, avl_dv = du / (h * w), dv / (h * w)
        radio = 0.5  #  If the value is too small, the color temperature develops to the pole

        valuekey = np.where((np.abs(u - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du)
                             | (np.abs(v - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv))
        num_y, yhistogram = np.zeros((h, w)), np.zeros(256)
        num_y[valuekey] = np.uint8(y[valuekey])
        yhistogram = np.bincount(np.uint8(num_y[valuekey].flatten()), minlength=256)
        ysum = len(valuekey[0])
        Y = 255
        num, key = 0, 0
        while Y >= 0:
            num += yhistogram[Y]
            if num > 0.1 * ysum:  #  Take the first 10% highlights as the calculated value, if the value is too large, the value is too small to adjust the amplitude
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(num_y > key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        num_rgb = len(sumkey[0])

        b0 = np.double(b) * int(max_y) / (sum_b / num_rgb)
        g0 = np.double(g) * int(max_y) / (sum_g / num_rgb)
        r0 = np.double(r) * int(max_y) / (sum_r / num_rgb)

        output_img = cv2.merge([b0, g0, r0])
    else:
        #  Default mean ---- Simple mean white balance method
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        #  Ask the gain of each channel
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        output_img = cv2.merge([b, g, r])
    output_img = np.uint8(np.clip(output_img, 0, 255))
    return output_img



'''

def percentile_whitebalance(image, percentile_value):
    fig, ax = plt.subplots(1,2, figsize=(12,6)) for channel, color in enumerate('rgb'):
        channel_values = image[:,:,channel]
        value = np.percentile(channel_values, percentile_value)
        ax[0].step(np.arange(256), 
                   np.bincount(channel_values.flatten(), 
                   minlength=256)*1.0 / channel_values.size, 
                   c=color)
        ax[0].set_xlim(0, 255)
        ax[0].axvline(value, ls='--', c=color)
        ax[0].text(value-70, .01+.012*channel, 
                   "{}_max_value = {}".format(color, value), 
                    weight='bold', fontsize=10)
    ax[0].set_xlabel('channel value')
    ax[0].set_ylabel('fraction of pixels');
    ax[0].set_title('Histogram of colors in RGB channels')    
    whitebalanced = img_as_ubyte(
            (image*1.0 / np.percentile(image, 
             percentile_value, axis=(0, 1))).clip(0, 1))
    ax[1].imshow(whitebalanced);
    ax[1].set_title('Whitebalanced Image')

    return ax

 '''

def IPOL_color_balance(img_input, img_output, cwd, Smin = 7, Smax = 7):

    coverted_input = img_input[:-5] + '.png'

    im1 = Image.open(img_input)
    im1.save(coverted_input)

    so_file = cwd + "/simplest_color_balance/balance"
    bashCommand = so_file + ' ' + 'rgb' + ' '+ str(Smin) + ' ' + str(Smax) + ' ' + coverted_input + ' ' + img_output

    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


def PIL_saturation(img_input, img_output):

    img = Image.open(img_input)
    new_image = ImageEnhance.Color(img)
    new_image.enhance(1.4).save(img_output)


def DW_C_WB(img_input, img_output):    

    img = cv2.imread(img_input)
    
    img = dynamic_threshold_white_balance(img)
    img = color_correction_of_image_analysis(img)
    img = WhiteBlance(img)
    cv2.imwrite(img_output, img)


def G_CC_WB(img_input, img_output):

    img = cv2.imread(img_input)

    img = gamma_trans(img)
    img = color_correction_of_image_analysis(img)
    img = WhiteBlance(img)
    cv2.imwrite(img_output, img)


def SCB_PILSATUR(img_input, img_output, cwd):

    IPOL_color_balance(img_input, img_output, cwd, Smin = 7, Smax = 7)
    PIL_saturation(img_output, img_output)


color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}


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

    for img_name in img_list:
        
        SCB_PILSATUR(img_name, 'results/simple_balance_color/' + img_name.split('/')[-1], os.getcwd())

        print('Process: ', img_name)

    print('done')