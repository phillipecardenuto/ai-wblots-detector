"""
Filters library.

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""

import numpy as np
from scipy.signal import convolve2d
from skimage import restoration
from PIL import Image
from typing import List
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def residuum(img: np.ndarray, 
             H = np.array([[0,    0.25, 0   ],
                           [0.25, 0   , 0.25],
                           [0,    0.25, 0   ]]),) -> np.ndarray:
    """
    Compute high-pass image residual
    :param img: np.ndarray, input image array
    :return: np.ndarray, high-pass image residual

    The function will apply a low-pass filter and calculated the residual
    from the original image.
    ## Default kernel  H used on Mandelli et al.
    -- Forensic Analysis of Synthetically Generated Western Blot Images.
     H = np.array([[0,    0.25, 0   ],
                   [0.25, 0   , 0.25],
                   [0,    0.25, 0   ]]),
   # Matthias kernel
    H = np.array([[-0.25, 0.5, -0.25],
                  [0.5  , 0.0, 0.5  ],
                  [-0.25, 0.5, -0.25]])
    # Mean
    H = np.ones((3,3), dtype=np.float32) / 8
    H[2, 2] = 0

    # Cross
    H = np.array([[1.0, -1.0],
                 [-1.0, 1.0]])
    """
    # Normalize image
    img = img.astype(np.float32)
    if np.max(img) > 1:
        img /= 255

    if img.ndim > 2:  # Keep only luminance component
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    # High-pass residual computation
    e = img - convolve2d(img, H, mode='same', boundary='symm')

    return e


    
def wavelets(img: np.ndarray, wavelet="db1") -> np.ndarray:
    """
    Apply wavelet transform to the image from skimage library.

    :param img: image to apply the filter.
    :param wavelet: str
        Wavelet to use. Default is 'db1'
    :return: image with the high-pass filter applied.
    """

    img = img.astype(np.float32)
    if np.max(img) > 1:
        img /= 255
    h_img = restoration.denoise_wavelet(img, wavelet=wavelet)
    e = img - h_img

    return e

def nlmeans(img: np.ndarray, patch_size=3,
            patch_distance=1, h=0.8 ) -> np.ndarray:
    """
    Apply non-local means denoising to the image, from skimage library.
    
    :param img: image to apply the filter.
    :param patch_size: int
        Size of the patch
    :param patch_distance: int
        Distance between patches
    :param h: float
        Filtering strength
    :param sigma: float
        Standard deviation of the noise
    :return: image with the high-pass filter applied.
    """

    img = img.astype(np.float32)
    if img.max() >= 1:
        img = img / 255.0
    sigma = restoration.estimate_sigma(img) 
    h_img = restoration.denoise_nl_means(img, h=h*sigma, 
                                             sigma=sigma,
                                             patch_size=patch_size,
                                             patch_distance=patch_distance,
                                             fast_mode=True)
    e = img - h_img

    return e

def gaussian_residuum(img: np.ndarray, sigma=1 ) -> np.ndarray:
    """
    Apply gaussian filter to the image and get its residual noise.
    
    :param img: image to apply the filter.
    :param sigma: float
        Kernel Sigma (default =1)
    :return: image with the high-pass filter applied.
    """

    img = img.astype(np.float32)
    if img.max() >= 1:
        img = img / 255.0
    h_img = gaussian_filter(img, sigma)
    e = img - h_img

    return e

def pmap(img: np.ndarray, lambda_=1, sigma=0.000001) -> \
    np.ndarray:
    """
    Creates a p-map from the image.
    It is possible to estimate sigma by using restoration.estimate_sigma()
    Default kernel values follows:

    Matthias Kirchner. 2008. Fast and reliable resampling detection by spectral 
    analysis of fixed linear predictor residue. In Proceedings of the 10th ACM 
    workshop on Multimedia and security.
    DOI https://doi.org/10.1145/1411328.1411333

    :param img: image to apply the filter.
    :param sigma: float

    """
    kernel = np.array([[0.25, 0.0, 0.25],
                       [ 0.0, 0, 0.0],
                       [0.25, 0.0, 0.25]])

    res = residuum(img, kernel)
    res = lambda_ * np.exp(-(res)**2 / sigma)
    return res