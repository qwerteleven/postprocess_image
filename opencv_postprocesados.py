import cv2
import numpy as np
import math
 

'''
# PIL enhancement



image = PIL.Image.open("path_to_image")

#increasing the brightness 20%
new_image = PIL.ImageEnhance.Brightness(image).enhance(1.2)

#increasing the contrast 20%
new_image = PIL.ImageEnhance.Contrast(image).enhance(1.2)


#convert pil.image to opencv (numpy.ndarray)
#need numpy library for this
cv_image = numpy.array(pil_image)

#convert opencv to pil.image

image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image)

import Image
import numpy as np
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
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

'''

'''
import cv2

#-----Reading the image-----------------------------------------------------
img = cv2.imread('Dog.jpg', 1)
cv2.imshow("img",img) 

#-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)

def mean_white_balance(img):
    """
    El primer método simple de balance de blancos promedio
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen de resultado del balance de blancos devueltos
    """
         # Leer imagen
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
         # Encuentra la ganancia ocupada por cada canal
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img

'''

def saturate(img, percentile):
    """Changes the scale of the image so that half of percentile at the low range
    becomes 0, half of percentile at the top range becomes 255.
    """

    if 2 != len(img.shape):
        raise ValueError("Expected an image with only one channel")

    # copy values
    channel = img[:, :].copy()
    flat = channel.ravel()

    # copy values and sort them
    sorted_values = np.sort(flat)

    # find points to clip
    max_index = len(sorted_values) - 1
    half_percent = percentile / 200
    low_value = sorted_values[math.floor(max_index * half_percent)]
    high_value = sorted_values[math.ceil(max_index * (1 - half_percent))]

    # saturate
    channel[channel < low_value] = low_value
    channel[channel > high_value] = high_value

    # scale the channel
    channel_norm = channel.copy()
    cv2.normalize(channel, channel_norm, 0, 255, cv2.NORM_MINMAX)

    return channel_norm

def adjust_gamma(img, gamma):
    """Build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.
    """

    # code from
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def adjust_brightness_with_gamma(gray_img, minimum_brightness, gamma_step=0.5):

    """Adjusts the brightness of an image by saturating the bottom and top
    percentiles, and changing the gamma until reaching the required brightness.
    """
    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    cols, rows = gray_img.shape
    changed = False
    old_brightness = np.sum(gray_img) / (255 * cols * rows)
    new_img = gray_img
    gamma = 1

    while True:
        brightness = np.sum(new_img) / (255 * cols * rows)
        if brightness >= minimum_brightness:
            break

        gamma += gamma_step
        new_img = adjust_gamma(gray_img, gamma = gamma)
        changed = True

    if changed:
        print("Old brightness: %3.3f, new brightness: %3.3f " %(old_brightness, brightness))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)

    return new_img

def main(filepath):

    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saturated = saturate(gray, 1)
    bright = adjust_brightness_with_gamma(saturated, minimum_brightness = 0.66)

 
def perfect_reflective_white_balance(img_input):
    """
         Balance de blancos de reflejo perfecto
         PASO 1: Calcule la suma de R \ G \ B para cada píxel
         PASO 2: Según el valor de R + G + B, calcular el valor del% de relación anterior como el umbral T del punto de referencia
         PASO 3: Para cada punto de la imagen, calcule el valor promedio de la suma acumulada de los componentes R \ G \ B de todos los puntos donde el valor R + G + B es mayor que T
         PASO 4: Cuantifique el píxel a [0,255] para cada punto
         Confiar en la selección del valor de relación y la imagen que no es blanca en el área más brillante no funciona bien.
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen de resultado del balance de blancos devueltos
    """
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    # for i in range(m):
    #     for j in range(n):
    #         sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
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
 
    # sum_b, sum_g, sum_r = 0, 0, 0
    # for i in range(m):
    #     for j in range(n):
    #         if sum_[i][j] >= key:
    #             sum_b += b[i][j]
    #             sum_g += g[i][j]
    #             sum_r += r[i][j]
    #             time = time + 1
    sum_b = b[sum_ >= key].sum()
    sum_g = g[sum_ >= key].sum()
    sum_r = r[sum_ >= key].sum()
    time = (sum_ >= key).sum()
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
    # maxvalue = 255
    # for i in range(m):
    #     for j in range(n):
    #         b = int(img[i][j][0]) * maxvalue / int(avg_b)
    #         g = int(img[i][j][1]) * maxvalue / int(avg_g)
    #         r = int(img[i][j][2]) * maxvalue / int(avg_r)
    #         if b > 255:
    #             b = 255
    #         if b < 0:
    #             b = 0
    #         if g > 255:
    #             g = 255
    #         if g < 0:
    #             g = 0
    #         if r > 255:
    #             r = 255
    #         if r < 0:
    #             r = 0
    #         img[i][j][0] = b
    #         img[i][j][1] = g
    #         img[i][j][2] = r
 
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
    """
         Hipótesis del mundo gris
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen de resultado del balance de blancos devueltos
    """
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
 
    # for i in range(len(Ba)):
    #     for j in range(len(Ba[0])):
    #         Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
    #         Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
    #         Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]
    Ba[Ba > 255] = 255
    Ga[Ga > 255] = 255
    Ra[Ra > 255] = 255
 
    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img
 
 
def color_correction_of_image_analysis(img):
    """
         Método de detección y corrección de color basado en análisis de imágenes
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen de resultado del balance de blancos devueltos
    """
 
    def detection(img):
        """Calcular el valor del reparto de color"""
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b, M_a, M_b = 0, 0, 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
        D = np.sqrt((np.square(d_a) + np.square(d_b)))
 
        for i in range(m):
            for j in range(n):
                M_a = np.abs(a[i][j] - d_a - 128) + M_a
                M_b = np.abs(b[i][j] - d_b - 128) + M_b
 
        M_a, M_b = M_a / (m * n), M_b / (m * n)
        M = np.sqrt((np.square(M_a) + np.square(M_b)))
        k = D / M
                 # print ('Valor de tinte de color:% f'% k)
        return
 
    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    # detection(img)
 
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    # sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    # max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(
    #     b[0][0]), int(g[0][0])
    #
    # for i in range(m):
    #     for j in range(n):
    #         I_r_2[i][j] = int(r[i][j] ** 2)
    #         I_b_2[i][j] = int(b[i][j] ** 2)
    #         sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
    #         sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
    #         sum_I_g = g[i][j] + sum_I_g
    #         sum_I_r = r[i][j] + sum_I_r
    #         sum_I_b = b[i][j] + sum_I_b
    #         if max_I_r < r[i][j]:
    #             max_I_r = r[i][j]
    #         if max_I_r_2 < I_r_2[i][j]:
    #             max_I_r_2 = I_r_2[i][j]
    #         if max_I_g < g[i][j]:
    #             max_I_g = g[i][j]
    #         if max_I_b_2 < I_b_2[i][j]:
    #             max_I_b_2 = I_b_2[i][j]
    #         if max_I_b < b[i][j]:
    #             max_I_b = b[i][j]
 
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
    # print(u_b, v_b, u_r, v_r)
 
    # b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    # for i in range(m):
    #     for j in range(n):
    #         b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
    #         g0[i][j] = g[i][j]
    #         # r0[i][j] = r[i][j]
    #         r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
    #         if r_point > 255:
    #             r0[i][j] = 255
    #         else:
    #             if r_point < 0:
    #                 r0[i][j] = 0
    #             else:
    #                 r0[i][j] = r_point
    #         if b_point > 255:
    #             b0[i][j] = 255
    #         else:
    #             if b_point < 0:
    #                 b0[i][j] = 0
    #             else:
    #                 b0[i][j] = b_point
 
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
    """
         Algoritmo de umbral dinámico
         El algoritmo se divide en dos pasos: detección del punto blanco y ajuste del punto blanco.
         Es solo que la detección del punto blanco no es lo mismo que el algoritmo de reflexión perfecta, que el punto más brillante es el punto blanco, pero está determinado por otra regla.
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen de resultado del balance de blancos devueltos
    """
 
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
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
 
    # for i in range(m):
    #     for j in range(n):
    #         sum_u = sum_u + u[i][j]
    #         sum_v = sum_v + v[i][j]
 
    sum_u = u.sum()
    sum_v = v.sum()
 
    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
 
    # for i in range(m):
    #     for j in range(n):
    #         du = du + np.abs(u[i][j] - avl_u)
    #         dv = dv + np.abs(v[i][j] - avl_v)
 
    du = (np.abs(u - avl_u)).sum()
    dv = (np.abs(v - avl_v)).sum()
 
    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5 # Si el valor es demasiado grande o demasiado pequeño, la temperatura del color se desarrollará a dos extremos
 
    # for i in range(m):
    #     for j in range(n):
    #         value = 0
    #         if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
    #                 v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
    #             value = 1
    #         else:
    #             value = 0
    #
    #         if value <= 0:
    #             continue
    #         num_y[i][j] = y[i][j]
    #         yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
    #         ysum += 1
 
    u_temp = np.abs(u.astype(np.float32) - (avl_u.astype(np.float32) + avl_du * con_num(avl_u))) < radio * avl_du
    v_temp = np.abs(v.astype(np.float32) - (avl_v.astype(np.float32) + avl_dv * con_num(avl_v))) < radio * avl_dv
    temp = u_temp | v_temp
    num_y[temp] = y[temp]
    # yhistogram = cv2.calcHist(num_y, 0, u_temp | v_temp, [256], [0, 256])
    # yhistogram[num_y[u_temp | v_temp].flatten().astype(np.int32)] += 1
    for i in range(m):
        for j in range(n):
            if temp[i][j] > 0:
                yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
    ysum = (temp).sum()
 
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num> 0.1 * ysum: # Toma el primer 10% de los puntos brillantes como valor calculado. Si el valor es demasiado grande, es fácil sobreexponer. Si el valor es demasiado pequeño, el rango de ajuste es pequeño
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
 
    # for i in range(m):
    #     for j in range(n):
    #         if num_y[i][j] > key:
    #             sum_r = sum_r + r[i][j]
    #             sum_g = sum_g + g[i][j]
    #             sum_b = sum_b + b[i][j]
    #             num_rgb += 1
 
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
 
    # for i in range(m):
    #     for j in range(n):
    #         b_point = int(b[i][j]) * int(max_y) / avl_b
    #         g_point = int(g[i][j]) * int(max_y) / avl_g
    #         r_point = int(r[i][j]) * int(max_y) / avl_r
    #         if b_point > 255:
    #             b[i][j] = 255
    #         else:
    #             if b_point < 0:
    #                 b[i][j] = 0
    #             else:
    #                 b[i][j] = b_point
    #         if g_point > 255:
    #             g[i][j] = 255
    #         else:
    #             if g_point < 0:
    #                 g[i][j] = 0
    #             else:
    #                 g[i][j] = g_point
    #         if r_point > 255:
    #             r[i][j] = 255
    #         else:
    #             if r_point < 0:
    #                 r[i][j] = 0
    #             else:
    #                 r[i][j] = r_point
 
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
    """
         corrección gamma
         Usar corrección de gamma adaptativa
         : param img: datos de imagen leídos por cv2.imread
         : retorno: los datos de imagen devueltos después de la corrección de gamma
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma = math.log10 (0.5) / math.log10 (mean / 255) # La fórmula calcula gamma
    gamma_table = [np.power (x / 255.0, gamma) * 255.0 for x in range(256)] # Crear una tabla de mapeo
    gamma_table = np.round (np.array (gamma_table)). astype (np.uint8) # El valor del color es un número entero
    return cv2.LUT (img, gamma_table) # Busque la tabla de colores de la imagen. Además, se puede diseñar un algoritmo adaptativo de acuerdo con el principio de homogeneización de la intensidad de la luz (color).
 
 
def contrast_image_correction(img):
    """
         Reproducción de Python de papel de corrección de imagen de contraste, tecnología HDR
         : param img: datos de imagen leídos por cv2.imread
         : return: los datos de imagen devueltos después de la corrección HDR
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    mv = cv2.split(img_yuv)
    img_y = mv[0].copy();
 
    # temp = img_y
    temp = cv2.bilateralFilter(mv[0], 9, 50, 50)
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         exp = np.power(2, (128 - (255 - temp[i][j])) / 128.0)
    #         temp[i][j] = int(255 * np.power(img_y[i][j] / 255.0, exp))
    #         # print(exp.dtype)
    # print(temp.dtype)
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
 
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         if (img_y[i][j] == 0):
    #             dst[i, j, :] = 0
    #         else:
    #             for k in range(3):
    #                 val = temp[i, j]/img_y[i, j]
    #                 val1 = int(img[i, j, k]) + int(img_y[i, j])
    #                 val2 = (val * val1+ img[i, j, k] - img_y[i, j]) / 2
    #                 dst[i, j, k] = int(val2)
    #             """
         # ERROR: El uso directo del siguiente método de cálculo hará que el valor se desborde, lo que resultará en resultados de cálculo incorrectos
    #             """
    # dst[i, j, 0] = (temp[i, j] * (img[i, j, 0] + img_y[i, j]) / img_y[i, j] + img[i, j, 0] - img_y[
    #     i, j]) / 2
    # dst[i, j, 1] = (temp[i, j] * (img[i, j, 1] + img_y[i, j]) / img_y[i, j] + img[i, j, 1] - img_y[
    #     i, j]) / 2
    # dst[i, j, 2] = (temp[i, j] * (img[i, j, 2] + img_y[i, j]) / img_y[i, j] + img[i, j, 2] - img_y[
    #     i, j]) / 2
 
    return dst


def WhiteBlance(img, mode=1):
    """White balance processing (default is 1 mean, 2 perfect reflection, 3 
    grayscale world, 4 based image analysis "and color correction, 5 dynamic threshold)
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

def percentile_whitebalance(image, percentile_value):fig, ax = plt.subplots(1,2, figsize=(12,6)) for channel, color in enumerate('rgb'):
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



def saturation_hsv(img):


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))
    # hsv[:,:,1] = greenMask 
    img[greenMask == 255] = (0, 255, 0)

    #blueMask = cv2.inRange(hsv, (90, 50, 70), (128, 255, 255))
    #hsv[:,:,1] = blueMask 

    #whiteMask = cv2.inRange(hsv, (0, 0, 231), (180, 18, 255))
    #hsv[:,:,1] = whiteMask 



    return img


 
if __name__ == '__main__':
    """
         img: imagen original
         img1: método de balance de blancos medio
         img2: reflejo perfecto
         img3: hipótesis del mundo en escala de grises
         img4: método de detección y corrección de color basado en el análisis de imágenes
         img5: algoritmo de umbral dinámico
         img6: corrección de gamma
         img7: corrección HDR
    """
 
    import time
 
    img = cv2.imread("GT12_Awesome.png")

    print(img.shape)
    # img = cv2.resize(img, (256, 512), cv2.INTER_LINEAR)
    img1 = WhiteBlance(img)
    img2 = perfect_reflective_white_balance(img)
    # img3 = gray_world_assumes_white_balance(img)

    img4 = color_correction_of_image_analysis(img1)
    img5 = dynamic_threshold_white_balance(img4)
    img6 = gamma_trans (img4) # transformación gamma
    img7 = contrast_image_correction(img4)
    # img8 = saturation_hsv(img5)

    cv2.imwrite("image1.png", img4)
    cv2.imwrite("image2.png", img5)
    cv2.imwrite("image3.png", img6)
    cv2.imwrite("image4.png", img7)
    cv2.imwrite("image49.png", img2)


    from PIL import Image,ImageEnhance
   
    img = Image.open("image2.png")

    new_image = ImageEnhance.Color(img)


    new_image.enhance(0.0).save('pil_saturation.png')

    # cv2.imshow("image2", img8)
    # img_stack = np.vstack([img, img1, img2, img3])
    # img_stack2 = np.vstack([img4, img5, img6, img7])
    # cv2.imshow("image1",img_stack)
    # cv2.imshow("image2",img_stack2)
