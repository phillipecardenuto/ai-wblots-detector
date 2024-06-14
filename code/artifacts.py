"""
Artifacts library.

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List



def glcm_feats(res: np.ndarray, levels=256, eps=1e-6) -> np.ndarray:
    """ Extract grayscale co-occurrences matrix features, similar as to
    S. Mandelli et al., "Forensic Analysis of Synthetically Generated Western
    Blot Images," in IEEE Access (2022). doi: 10.1109/ACCESS.2022.3179116

    :param res: Image residual
                Input image must be an integer image.
    :param levels: Number of gray levels (default 256)
    :return: np.array of features
    """
    # Assert image is in uint8
    if res.dtype != np.uint8:
        if np.max(res) <= 1:
            res -= np.min(res)
            res /= (np.max(res)+eps)
            res *= 255
        res[res < 0] = 0
        res[res > 255] = 255
        res = res.astype(np.uint8)  # cast the type if it isn't provided as integer
    # Compute the GLCM
    glcm = graycomatrix(res,
                      levels=levels,
                      distances=[4, 8, 16, 32],
                      angles=[0, np.pi/2],
                      symmetric=True,
                      normed=True)
    
    # Compute the GCLM feats
    f_c = graycoprops(glcm, prop='contrast')
    # homogeneity
    f_h = graycoprops(glcm, prop='homogeneity')
    # dissimilarity
    f_d = graycoprops(glcm, prop='dissimilarity')
    # energy
    f_e = graycoprops(glcm, prop='energy')
    # correlation
    f_ro = graycoprops(glcm, prop='correlation')


    feats = np.array([f_c, f_h, f_d, f_e, f_ro]).flatten()
    #  features
    return feats


def image_patches(img: np.ndarray, patch_height=64, patch_width=64) -> np.ndarray:
    """
    Extract non-overlap patches from an image.

    :param img: Grayscale image
                Input should be a grayscale image
    :param patch_height: Height of the patch
    :param patch_width: Width of the patch
    :return: np.array of patches
    """
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]

    # Calculate the number of patches along height and width
    num_patches_y = image_height // patch_height
    num_patches_x = image_width // patch_width

    # List to hold the patches
    patches = []

    # Iterate over the image to extract patches
    for y in range(0, num_patches_y * patch_height, patch_height):
        for x in range(0, num_patches_x * patch_width, patch_width):
            # Extract the patch
            patch = img[y:y + patch_height, x:x + patch_width]
            # only append if the patch is the correct size
            if patch.shape == (patch_height, patch_width):
                patches.append(patch)
    return np.array(patches)

def dft(signal) -> np.ndarray:
    """
    Perform a Discrete Fourier Transform on a signal
    :param signal: 2D numpy array
    return mag: 2D numpy array
    """
    # Calculate the fourier transform for each patch
    signal = np.fft.fft2(signal)

    signal =  np.fft.fftshift(signal)

    # get the magnitude of the fourier transform
    mag = np.abs(signal)

    # get the mean of the magnitude of the fourier transform
    return mag

def _peaks_location(height, width):
    """
    Locate the peaks in the 2D Fourier transform of the image.
    """

    locs_x = list(range(width//2, width, width//8)) + [width-1]
    locs_y = list(range(0, height, height//8)) + [height-1]
    locs = []
    for x in locs_x:
        for y in locs_y:
            locs.append((x, y))

    return locs

def fft_peak_feats( res: np.ndarray) -> np.ndarray:
    """
    Calculates the peaks features from the Fourier transform of the image.
    Similar to Q. Bammey, "Synthbuster: Towards Detection of Diffusion Model 
    Generated Images,".
    :param res: Image residual
    :return: peaks
    """
    spec = dft(res)
    spec = spec / (res.shape[0]*res.shape[1]) # Normalizing as in the paper
    locs = _peaks_location(height=spec.shape[0], 
                                            width=spec.shape[1])
    peaks = [spec[loc] for loc in locs]

    peaks = np.array(peaks)
    return peaks 

def patch_fft_peak_feats( res: np.ndarray, patch_size=64,) -> np.ndarray:
    """
    Compute the combined Patched FFT-PEAKS upon the residuum provided from an image.
    Make sure to input the residuum not the image.

    :param res: Image residual (calculate the residuum before using this function)
    :return: np.array with the peaks in the 2D Fourier transform of the image.
    """

    assert res.shape[0] >= patch_size, "Image height must be greater than patch size"
    assert res.shape[1] >= patch_size, "Image width must be greater than patch size"

    # Compute the 2D Fourier transform of the image
    patches = image_patches(res, patch_height=patch_size, patch_width=patch_size)
    patches_fourier = [dft(patch) for patch in patches]
    # Normalize patches dft
    for i, _ in enumerate(patches_fourier):
        patch_h, patch_w = patches_fourier[i].shape
        patches_fourier[i] = patches_fourier[i] / (patch_h * patch_w)

    # Combine patches
    patches_combined = np.array(patches_fourier).mean(axis=0)

    locs = _peaks_location(patches_combined.shape[0], patches_combined.shape[1])
    peaks = [patches_combined[loc] for loc in locs]
    peaks = np.array(peaks)

    return peaks

def fourier_glcm(res: np.ndarray, eps=1e-6) -> np.ndarray:	
    """
    Calculates the GLCM features from the Fourier transform of the co-occurrence matrix.
    :param res: Image residual
    :return: np.array of features
    """
    
    if res.dtype != np.uint8:
        if np.max(res) <= 1:
            res -= np.min(res)
            res /= (np.max(res)+eps)
            res *= 255
        res[res < 0] = 0
        res[res > 255] = 255
        res = res.astype(np.uint8)  # cast the type if it isn't provided as integer
    glcm = graycomatrix(res,
                        levels=256,
                        distances=[4, 8, 16, 32],
                        angles=[0, np.pi/2],
                        symmetric=True,
                        normed=True)
    # sum all directions and angles
    for i in range(4):
        for j in range(2):
            feat = dft(glcm[:, :, i, j])
            feat = feat / (256 * 256) # normalize feat by size of GLCM matrix
            glcm[:, :, i, j] = feat 

    f_c = graycoprops(glcm, prop='contrast')
    # homogeneity
    f_h = graycoprops(glcm, prop='homogeneity')
    # dissimilarity
    f_d = graycoprops(glcm, prop='dissimilarity')
    # energy
    f_e = graycoprops(glcm, prop='energy')
    # correlation
    f_ro = graycoprops(glcm, prop='correlation')

    feats = np.array([f_c, f_h, f_d, f_e, f_ro]).flatten()
    return feats