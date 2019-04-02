import math
import numpy as np
import cv2

def fitting_rectangle(*points):
    t = float('inf')
    l = float('inf')
    r = float('-inf')
    b = float('-inf')
    for x, y in points:
        if x > r:
            r = x
        if y > b:
            b = y
        if y < t:
            t = y
        if x < l:
            l = x
    l = int(math.floor(l))
    t = int(math.floor(t))
    return (l, t), (int(math.ceil(r - l)), int(math.ceil(b - t)))


def im_pst(base, img, shift):
    h, w = img.shape[:2]
    x, y = shift
    dest_slice = np.s_[y:y + h, x:x + w]
    dest = base[dest_slice]
    dest = cv2.add(cv2.bitwise_and(dest, dest, mask=(255 - img[..., 3])), img)
    base[dest_slice] = dest
    return


def colinfo(lab_image, mask=None):
    lab_image = lab_image.reshape(-1, lab_image.shape[-1]) if (mask is None) else lab_image[mask.nonzero()]
    return lab_image.mean(axis=0), lab_image.std(axis=0)
