import cv2
import numpy as np


def rgb2gray(rgb):
    """ Convert an RGB numpy image to grayscale
    Params:
        rgb: rgb image as numpy array with last axis size 3
    Returns:
        gray: numpy array with same shape as rgb but last axis size 1
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def preprocess_img(img):
    """ Run preprocessing on camera frames - ensure they are in the
    right range and grayscale.
    
    Params:
        img: numpy array of rank 3 but arbitrary shape
    Returns:
        preproc_img: numpy array with same shape as img but last axis
            ensured to be size 1 if not already
    """
    assert img.min() >= 0.
    assert img.max() <= 255.

    # convert to (0-255) range if needed
    if img.max() <= 1.:
        img *= 255.

    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = rgb2gray(img)

    return img.astype('uint8')


def mat2euler(matrix):
    """ Convert a rotation matrix into Euler angles.
    Assuming x, y, z ordering.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    
    cy = np.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0])
    if cy > 1e-9:
        ax = np.atan2( M[2, 1],  M[2, 2])
        ay = np.atan2(-M[2, 0],  cy)
        az = np.atan2( M[1, 0],  M[0, 0])
    else:
        ax = np.atan2(-M[1, 2],  M[1, 1])
        ay = np.atan2(-M[2, 0],  cy)
        az = 0.0

    return ax, ay, az


def euler2rvec(a0, a1, a2):
    """ convert euler angles to rodrigues vector """
    return np.squeeze(cv2.Rodrigues(euler2mat(a0, a1, a2))[0])


def rvec2euler(rvec):
    """ convert rodrigues vector into euler angles """
    return mat2euler(cv2.Rodrigues(rvec)[0])


def euler2mat(ai, aj, ak):
    """ convert euler angles to a rotation matrix """
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    
    M[0, 0] = cj*ck
    M[0, 1] = sj*sc-cs
    M[0, 2] = sj*cc+ss
    M[1, 0] = cj*sk
    M[1, 1] = sj*ss+cc
    M[1, 2] = sj*cs-sc
    M[2, 0] = -sj
    M[2, 1] = cj*si
    M[2, 2] = cj*ci

    return M