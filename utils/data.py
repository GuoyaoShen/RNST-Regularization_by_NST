import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import torch



def load_img(path):
    '''
    Load .dcm file and return the image.
    :param path: path of the .dcm file.
    :return: image, np array.
    '''
    ds = dicom.dcmread(path)
    img = np.asarray(ds.pixel_array)
    return img


def error_map(img1, img2):
    '''
    Get the error map of two images.
    '''
    return abs(img1 - img2)
