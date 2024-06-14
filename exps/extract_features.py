"""
This script extracts features from the dataset using the following methods:
- FFT Peaks
- Patch FFT Peaks
- GLCM
- FFT-GLCM
- DINOv1
- DINOv2
- CLIP

The hand-crafted FFT Peaks, Patch FFT Peaks, GLCM, and FFT-GLCM features are
extracted in parallel fashion using the thread_map function from the tqdm library.

The DINO and CLIP features uses GPU if available for faster processing.

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""


import os
import numpy as np
from typing import List
import sys
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
sys.path.append('../code')
# Dataset
from dataset import Dataset
from utils import load_img

# Features
from artifacts import fft_peak_feats, patch_fft_peak_feats
from artifacts import glcm_feats, fourier_glcm

# Filters
from filters import residuum, nlmeans, gaussian_residuum, pmap


import argparse
import torch
torch.backends.cudnn.deterministic = True
from transformers import AutoImageProcessor, AutoModel
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from dataset")
    parser.add_argument("--feat", type=str, required=True, help="Feature to extract")
    return parser.parse_args()


SEED = 0
EPS = 1e-06
np.random.seed(SEED)

# DATASET
# --------
dataset = Dataset(classes=["real", "cyclegan",  "ddpm", "stylegan2ada", 
                                "pix2pix", "sdxl", "figshare"], seed=SEED)
dataset = dataset.dataset

# FILTERS 
# ----------------
mean_filter = np.ones((3,3), dtype=np.float32) / 8
mean_filter[2, 2] = 0
filters = { 
        "gaussian": gaussian_residuum,
        "access": np.array([[0,    0.25, 0   ],
                            [0.25, 0   , 0.25],
                            [0,    0.25, 0   ]]),
        "matthias": np.array([[-0.25, 0.5, -0.25],
                            [0.5  , 0.0, 0.5  ],
                            [-0.25, 0.5, -0.25]]),
        "mean": mean_filter,
        "cross": np.array([[1.0, -1.0],
                            [-1.0, 1.0]]),
        "nlmeans": nlmeans,
        "pmap": pmap,
        "none": None,
}

def extract_fft_peaks(dataset):
    fft_feats = dataset.copy()
    for filter_name, filter in filters.items():
        print(f"Processing {filter_name}")
        def profile_parallel(item) -> List[float]:
            img_path = item["path"]
            img = load_img(img_path, crop_size=256)
            if filter_name in ["nlmeans", "gaussian", "pmap"]:
                res = filters[filter_name](img)
            elif filter_name == "none":
                res = img
            else:
                res = residuum(img, H=filters[filter_name])
            peaks = fft_peak_feats(res)
            return peaks 
    
        fft_feats[f"{filter_name}"] = thread_map(profile_parallel, dataset.to_records(), max_workers=45)
    fft_feats.to_pickle("fft_peak_feats.pkl")

def extract_patch_fft_peak(dataset):
    fft_feats = dataset.copy()
    for filter_name, filter in filters.items():
        print(f"Processing {filter_name}")
        def profile_parallel(item) -> List[float]:
            img_path = item["path"]
            img = load_img(img_path, crop_size=256)
            if filter_name in ["nlmeans", "gaussian", "pmap"]:
                res = filters[filter_name](img)
            elif filter_name == "none":
                res = img
            else:
                res = residuum(img, H=filters[filter_name])
            peaks = patch_fft_peak_feats(res, patch_size=64)
            return peaks 
    
        fft_feats[f"{filter_name}"] = thread_map(profile_parallel, dataset.to_records(), max_workers=45)
    fft_feats.to_pickle("patch_fft_peak_feats.pkl")

def extract_glcm(dataset):
    glcm_features= dataset.copy()
    for filter_name, filter in filters.items():
        print(f"Processing {filter_name}")
        def profile_parallel(item) -> List[float]:
            img_path = item["path"]
            img = load_img(img_path, crop_size=256)
            if filter_name in ["nlmeans", "gaussian", "pmap"]:
                res = filters[filter_name](img)
            elif filter_name == "none":
                res = img
            else:
                res = residuum(img, H=filters[filter_name])
            res  = (res - np.min(res)) / ((np.max(res) - np.min(res) + EPS))
            feats = glcm_feats(res)
            feats = np.array(feats)
            return feats
    
        glcm_features[f"{filter_name}"] = thread_map(profile_parallel, dataset.to_records(), max_workers=45)
    glcm_features.to_pickle("glcm_feats.pkl")

def extract_fft_glcm(dataset):
    fft_glcm_feats = dataset.copy()
    for filter_name, filter in filters.items():
        print(f"Processing {filter_name}")
        def profile_parallel(item) -> List[float]:
            img_path = item["path"]
            img = load_img(img_path, crop_size=256)
            if filter_name in ["nlmeans", "gaussian", "pmap"]:
                res = filters[filter_name](img)
            elif filter_name == "none":
                res = img
            else:
                res = residuum(img, H=filters[filter_name])
            res  = (res - np.min(res)) / ((np.max(res) - np.min(res) + EPS))
            feats = fourier_glcm(res)
            feats = np.array(feats)
            return feats
        
        fft_glcm_feats[f"{filter_name}"] = thread_map(profile_parallel, dataset.to_records(), max_workers=45)
    fft_glcm_feats.to_pickle("fft_glcm_feats.pkl")


# DINO RELATED FEATS EXTRACTION
# START ------------------------
# load model

def load_dino_model(version="dinoV1"):
    """
    Load DINO model and processor.
    :param version: DINO version to load. Options: dinoV1, dinoV2
    :return: DINO processor and model
    """

    if version == "dinoV1":
        processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16',  cache_dir=CACHE_DIR)
        model = ViTModel.from_pretrained('facebook/dino-vitb16',  cache_dir=CACHE_DIR)
    elif version == "dinoV2":
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)
    else:
        raise ValueError("Invalid version")
    return processor, model

def dino_feats(image: np.ndarray, 
             processor, model,

           ) -> np.ndarray:
    """
    Extract features using DINO model.
    Use load_dino_model to get the processor and model.

    :param image: Input image.
            In case the input is a grayscale image, it will be concatenated to 
            form a 3-channel image.
    :param processor: Dino processor
    :param model: Dino model
    :return: np.array with the features

    """
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    with torch.no_grad():
        do_rescale = True if image.max() > 1 else False
        inputs = processor(images=image,
                           return_tensors="pt",
                           do_rescale=do_rescale)
        inputs.to(device)
        outputs = model(**inputs)[0]
        feat = outputs.squeeze()[0].detach().cpu().numpy()
    return feat

def dinofeat(dataset, filter):
    feats = []
    for img_path in tqdm(dataset.path.values):
        img = load_img(img_path, crop_size=256)
        if filter in ["nlmeans", "gaussian", "pmap"]:
            res = filters[filter](img)
        elif filter == "none":
            res = img
        else:
            res = residuum(img, H=filters[filter])
        res  = (res - np.min(res)) / ((np.max(res) - np.min(res) + EPS))
        feats.append(dino_feats(res, dino_processor, dino_model))
    return feats

# DINO RELATED FEATS EXTRACTION
# END ---------------------------


def extract_dino(dataset, version="v1"):
    dino_feats = dataset.copy()
    for filter_name in filters.keys():
        print(f"Processing {filter_name}")
        dino_feats[f"{filter_name}"] = dinofeat(dino_feats, filter_name)
    dino_feats.to_pickle(f"dino{version}_feats.pkl")


# CLIP RELATED FEATS
# START ---------------------------
# Load the CLIP model
def load_clip_model():
    """
    Load CLIP model and processor.
    :return: CLIP processor and model
    """
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=CACHE_DIR)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir=CACHE_DIR)
    return processor, model

def clip_feats(image: np.ndarray, 
               processor, model,
              ) -> np.ndarray:
    """
    Extract features using CLIP model.
    Use load_clip_model to get the processor and model.

    :param image: Input image.
            In case the input is a grayscale image, it will be concatenated to 
            form a 3-channel image.
    :param processor: CLIP processor
    :param model: CLIP model
    :return: np.array with the features

    """
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    with torch.no_grad():
        inputs = processor(images=image,
                           return_tensors="pt")

        inputs.to(device)
        outputs = model.get_image_features(**inputs)
        feat = outputs.squeeze().detach().cpu().numpy()
    return feat

def clipfeat(dataset, filter, eps=1e-06):
    feats = []
    for img_path in tqdm(dataset.path.values):
        img = load_img(img_path, crop_size=256)
        if filter in ["nlmeans", "gaussian", "pmap"]:
            res = filters[filter](img)
        elif filter == "none":
            res = img
        else:
            res = residuum(img, H=filters[filter])
        res  = (res - np.min(res)) / ((np.max(res) - np.min(res) + eps))
        feats.append(clip_feats(res, clip_processor, clip_model))
    return feats
# CLIP RELATED FEATS EXTRACTION
# END ---------------------------

def extract_clip(dataset):
    clip_feats = dataset.copy()
    for filter_name in filters.keys():
        print(f"Processing {filter_name}")
        clip_feats[f"{filter_name}"] = clipfeat(clip_feats, filter_name)
    clip_feats.to_pickle(f"clip_feats.pkl")


# Parse arg
args = parse_args()
feat = args.feat
if feat == "fft_peaks":
    print("Extracting FFT-PEAKS features")
    extract_fft_peaks(dataset)
elif feat == "patch_fft_peaks":
    print("Extracting Patch FFT-PEAKS features")
    extract_patch_fft_peak(dataset)
elif feat == "glcm":
    print("Extracting GLCM features")
    extract_glcm(dataset)
elif feat =="fft_glcm":
    print("Extracting FFT-GLCM features")
    extract_fft_glcm(dataset)
elif feat == "dino":
    print("Extracting DINOv1 features")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dino_processor, dino_model = load_dino_model(version="dinoV1")
    dino_model.to(device)
    extract_dino(dataset)
elif feat == "dinov2":
    print("Extracting DINOv2 features")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dino_processor, dino_model = load_dino_model(version="dinoV2")
    dino_model.to(device)
    extract_dino(dataset, version="v2")
elif feat == "clip":
    print("Extracting CLIP features")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_processor, clip_model = load_clip_model()
    clip_model.to(device)
    extract_clip(dataset)
else:
    raise ValueError("Invalid feature to extract")

print(f"Extraction of {feat} features completed.")

