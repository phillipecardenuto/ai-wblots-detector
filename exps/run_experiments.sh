#!/bin/bash
# Use this script just after extracting the features using extract_feats.sh


# 1. Run closed set experiments
python closed-set.py --split xval

# 2. Run open set experiments
python openset.py --split xdomain

# 3. Run attribution experiments
python attribution.py --split xdomain

echo "Experiments completed"