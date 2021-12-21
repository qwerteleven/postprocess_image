import cv2 as cv
import numpy as np

# need a 2 or more images with diferents exposuress

def hdr_postprocess(img_fn):

    # Loading exposure images into a list
    img_list = [cv.imread(fn) for fn in img_fn]

    exposure_times = np.array([100, 250, 500], dtype=np.float32)

    # Merge exposures to HDR image
    merge_debevec = cv.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
    merge_robertson = cv.createMergeRobertson()
    hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

    # Tonemap HDR image
    tonemap1 = cv.createTonemap(gamma=2.2)
    res_debevec = tonemap1.process(hdr_debevec.copy())

    # Exposure fusion using Mertens
    merge_mertens = cv.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)

    # Convert datatype to 8-bit and save
    res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

    return (res_debevec_8bit, res_mertens_8bit)
