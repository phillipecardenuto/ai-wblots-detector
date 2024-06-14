
# Load Dataset
from pathlib import Path
from glob import glob
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset():
    """ Creates a train and test dataset."""
    DATAPATH = "../data/access-2022/western_blots_dataset/"
    FIGSHARE_DATAPATH_FOLD1 = "../data/fold1/"
    FIGSHARE_DATAPATH_FOLD2 = "../data/fold2/"


    def __init__(self, n_samples=-1, seed=42,  train_size=0.5,
                classes=["real", "cyclegan",  "ddpm", 
                         "stylegan2ada", "pix2pix"],
                 ):

        """
        Creates a dataset in pandas dataframe format.
        :param: n_samples: number of samples to load from each class
        :param: seed: random seed
        :param: train_size: size of the training dataset
        :param: classes: list of classes to load
                classes could be: ["real", "cyclegan",  "ddpm", "stylegan2ada", 
                                  "pix2pix", "sdxl", "dalle3", "figshare"]

        """

        # set np seed
        np.random.seed(seed)
        classes = self._validate_classes(classes)
        # load dataset
        self.dataset = self.load_dataset(n_samples, classes)

        self.train, self.test = train_test_split(self.dataset, 
                                                 train_size=train_size, 
                                                 stratify=self.dataset['type'], 
                                                 random_state=seed)


    def load_dataset(self, n_samples, classes):

        # Make a balanced dataset with 6000 samples
        if n_samples == -1:
            n_samples = 6000

        # Read data
        dataset = []
        types_ = []
        if "real" in classes:
            real_dataset = sorted(glob(f"{Dataset.DATAPATH}/real/*.png")[:n_samples])
            dataset += real_dataset
            types_ += ["real"] * len(real_dataset)
        if "cyclegan" in classes:
            cyclegan_dataset = sorted(glob(f"{Dataset.DATAPATH}/synth/cyclegan/*.png")[:n_samples])
            dataset += cyclegan_dataset
            types_ += ["cyclegan"] * len(cyclegan_dataset)
        if "ddpm" in classes:
            ddpm_dataset = sorted(glob(f"{Dataset.DATAPATH}/synth/ddpm/*.png")[:n_samples])
            dataset += ddpm_dataset
            types_ += ["ddpm"] * len(ddpm_dataset)
        if "pix2pix" in classes:
            pix2pix_dataset = sorted(glob(f"{Dataset.DATAPATH}/synth/pix2pix/*.png")[:n_samples])
            dataset += pix2pix_dataset
            types_ += ["pix2pix"] * len(pix2pix_dataset)
        if "stylegan2ada" in classes:
            stylegan2ada_dataset = sorted(glob(f"{Dataset.DATAPATH}/synth/stylegan2ada/*.png")[:n_samples])
            dataset += stylegan2ada_dataset
            types_ += ["stylegan2ada"] * len(stylegan2ada_dataset)
       
        if "figshare" in classes:
            figshare_dataset_fold1 = sorted(glob(f"{Dataset.FIGSHARE_DATAPATH_FOLD1}/*"))
            samples = min(len(figshare_dataset_fold1), n_samples//2)

            figshare_dataset_fold1 = np.random.choice(figshare_dataset_fold1, size=samples, replace=False)

            figshare_dataset_fold2 = sorted(glob(f"{Dataset.FIGSHARE_DATAPATH_FOLD2}/*"))
            figshare_dataset_fold2 = np.random.choice(figshare_dataset_fold2, size=samples, replace=False)

            figshare_dataset = list(figshare_dataset_fold1) + list(figshare_dataset_fold2)
            dataset += figshare_dataset
            types_ += ["figshare"] * len(figshare_dataset)
        
        # Create dataframe
        dataset = pd.DataFrame({"path": dataset, "type": types_})
        dataset.sort_values(by="type", inplace=True)
        dataset.sort_values(by=["type", "path"], inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def _validate_classes(self, classes):
        valid_classes = ["real", "cyclegan",  "ddpm", 
                         "stylegan2ada", "pix2pix", "figshare"]
        for c in classes:
            assert type(c) == str, "Classes must be strings"
        
        classes = [c.lower() for c in classes]
        classes = list(set(classes))

        for c in classes:
            if c not in valid_classes:
                raise ValueError(f"Invalid class {c}. Valid classes are {valid_classes}")
        return classes
  