#!/bin/bash
for feat in fft_peaks patch_fft_peaks glcm fft_glcm dinov1 dinov2 clip
do
    echo "Extracting features for $feat"
    python extract_features.py --feat $feat &
done
