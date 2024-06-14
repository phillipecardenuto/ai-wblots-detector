"""
Image Utils library.

Authors: Sara Mandelli and Joao Phillipe Cardenuto.
Date: April, 2024
"""

from PIL import Image, ImageFilter
import cv2
import tempfile
import numpy as np

def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)
  
def load_img(img_path: str, jpeg_quality:int=100, crop_size:int=0, blur_factor=0,
             min_height=256) -> np.ndarray:
    """
    Load an image from a path and apply a jpeg compression to it.
    :param img_path: path to image
    :param jpeg_quality: jpeg quality to apply
    :param crop_size: size of the crop
    :param blur_factor: factor to apply gaussian blur
    :param min_height: assert image have a minimum height of 256
    :return: image as numpy array
    """

    assert jpeg_quality >= 0 and jpeg_quality <= 100, "jpeg_quality must be between 0 and 100"
    assert blur_factor >= 0, "blur_factor must be greater than 0"

    img = Image.open(img_path)
    if blur_factor:
        img = img.filter(ImageFilter.GaussianBlur(blur_factor))

    img = np.array(img)

    # assert image height is min_height
    if img.shape[0] < min_height:
        heigh, width = img.shape[:2]
        new_width = int(width * min_height / heigh)
        img = cv2.resize(img, (new_width, min_height))
    
    # Use tempfile.NamedTemporaryFile to create a unique temp file for each call
    if jpeg_quality < 100:
        # convert image to uint8
        img = img.astype(np.uint8)
        timg = Image.fromarray(img)

        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpeg') as tmp_file:
            temp_filename = tmp_file.name
            # Save the image with the desired quality to the temporary file
            timg.save(temp_filename, quality=jpeg_quality)
            img = np.array(Image.open(temp_filename))
            img = rgb2gray(img)
    
    if crop_size:
        img = center_crop(img, crop_size)


    img = rgb2gray(img)
    

    return img

def center_crop(img: np.ndarray, size: int=256) -> np.ndarray:
    """
    Perform crop of the image to the center.
    :param img: image to crop
    :param size: size of the crop
    :return: cropped image
    """

    # get center of image
    x = img.shape[0] // 2
    y = img.shape[1] // 2

    # get the center of the image
    z = size//2

    # Get centered x0 and x1
    x0 = max(0, x-z)
    x1 = min(img.shape[0], x+z)
    if x1 == img.shape[0]:
        x0 = max(0, x1-z*2)
    if x0 == 0:
        x1 = min(img.shape[0], x0+z*2)

    # Get centered y0 and y1
    y0 = max(0, y-z)
    y1 = min(img.shape[1], y+z)
    if y1 == img.shape[1]:
        y0 = max(0, y1-z*2)
    if y0 == 0:
        y1 = min(img.shape[1], y0+z*2)

    # crop the image
    img = img[x0:x1, y0:y1]

    # Check if the image is square
    if img.shape[0] != img.shape[1]:
        shape = min(img.shape[0], img.shape[1])
        img = center_crop(img, shape)

    return img