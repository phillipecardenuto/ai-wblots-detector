"""
Experiment Util Library

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

def cross_val_split(data, 
                data_type=["cyclegan", "ddpm", "stylegan2ada", 
                           "real", "pix2pix"],
                SEED=0
     ):
    """
    Divide dataframe in two splits
    """
    temp_data = data.copy()
    temp_data = temp_data[temp_data["type"].isin(data_type)]
    split1, split2 = train_test_split(temp_data, 
                                      train_size=0.5, 
                                      stratify=temp_data['type'], 
                                      random_state=SEED)
    return split1, split2

def cross_domain_split(data,
                       data_type=["cyclegan", "ddpm", "stylegan2ada", 
                                  "real", "pix2pix", "figshare"],
                       SEED=0
                       ):
    # make split1 contain all figshare data and split2 all (Mandelli's) real data
    if "figshare" in data_type:
        figshare = data[data["type"]=="figshare"].copy()
        figshare = figshare.sample(frac=0.5, random_state=SEED, replace=False)
        real = data[data["type"]=="real"].copy()
        real = real.sample(frac=0.5, random_state=SEED, replace=False)

  
    temp_data = data.drop(data[(data["type"]=="figshare") | (data["type"]=="real")].index)
    split1, split2 = train_test_split(temp_data, 
                                      train_size=0.5, 
                                      stratify=temp_data['type'], 
                                      random_state=SEED)
    
    if "figshare" in data_type:
        figshare["type"] = "real"
        split1 = pd.concat([split1, figshare])
        split2 = pd.concat([split2, real])
    return split1, split2

def read_features():
    assert os.path.exists("fft_peak_feats.pkl"), "Please Extract the Fourier Feats using extract_features.py and save in this same directory."
    assert os.path.exists("fft_glcm_feats.pkl"), "Please Extract the Fourier-GLCM Feats using extract_features.py and save in this same directory."
    assert os.path.exists("glcm_feats.pkl"), "Please Extract the GLCM Feats using extract_features.py and save in this same directory."
    assert os.path.exists("dinov1_feats.pkl"), "Please Extract the DinoV1 Feats using extract_features.py and save in this same directory."
    assert os.path.exists("dinov2_feats.pkl"), "Please Extract the DinoV2 Feats using extract_features.py and save in this same directory."
    assert os.path.exists("clip_feats.pkl"), "Please Extract the Clip Feats using extract_features.py and save in this same directory."
    assert os.path.exists("patch_fft_peak_feats.pkl"), "Please Extract the Patch Fourier Feats using extract_features.py and save in this same directory."

    fft_feats = pd.read_pickle("fft_peak_feats.pkl")
    fft_glcm_feats = pd.read_pickle("fft_glcm_feats.pkl")
    glcm_feats = pd.read_pickle("glcm_feats.pkl")
    dinov1_feats = pd.read_pickle("dinov1_feats.pkl")
    dinov2_feats = pd.read_pickle("dinov2_feats.pkl")
    clip_feats = pd.read_pickle("clip_feats.pkl")
    patch_fft_feats = pd.read_pickle("patch_fft_peak_feats.pkl")

    feats = {
            "FFT-PEAKS": fft_feats,
            "FFT-GLCM": fft_glcm_feats,
            "GLCM": glcm_feats,
            "DINOV1": dinov1_feats,
            "DINOV2": dinov2_feats,
            "CLIP": clip_feats,
            "PATCH-FFT-PEAKS": patch_fft_feats,
             }
    return feats