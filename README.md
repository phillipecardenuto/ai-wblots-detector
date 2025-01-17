# Exploring Explainable Features for AI Western Blot Detection

**TL;DR:** This repository provides an analysis and implements explainable features to detect AI-generated Western blots. It also explores how to attribute the AI-generated Western blots to the models that produced them.

The accessibility of generative AI models has enabled non-experts to create their own data. While these models have numerous beneficial applications, they pose a significant concern in the scientific community regarding the generation of fake images and texts.

Our work aims to detect fake scientific images generated by advanced AI models using explainable solutions. In addition to proposing detection methods, we evaluate existing and proposed solutions to trace the source of AI-generated images. This attribution task is crucial for identifying the systematic production of AI-generated images, such as by paper mills. Understanding which models are used by these organizations can help track individuals and entities involved in scientific misconduct.

All proposed solutions are organized in the **code** directory.



### Dataset

We are releasing all datasets used in this research. Please refer to the README file in the **data** directory to access the datasets.

### Quick Run

To reproduce our experiments and figures (as detailed in our article), follow these instructions:

Run the following commands in the /exps directory.

1. Install requirements:

   ```
   pip install -r requirements.txt
   ```

2. Extract features for all datasets:

   ```
   ./extract_feats.sh
   ```

   After running this command, you should see the following pickle files in the experiment directory: `dinov1`, `dinov2`, `clip`, `fft_peak`, `patch_fft_peak`, `glcm`, and `fft_glcm` .

3. Run experiments:

   ````
   run_experiments.sh
   ````

   After running this command, you should see `closed_set`, `open_set`, and `attribution` CSV files containing the results for each type of residuum extraction and each feature explored during our experiments.

### Citation
```
@inproceedings{cardenuto2024Explainable,
  author={Cardenuto, João P. and Mandelli, Sara and Moreira, Daniel and Bestagini, Paolo and Delp, Edward and Rocha, Anderson},
  booktitle={2024 IEEE International Workshop on Information Forensics and Security (WIFS)}, 
  title={Explainable Artifacts for Synthetic Western Blot Source Attribution}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Forensics;Conferences;Paper mills;Closed box;Organizations;Generative adversarial networks;Diffusion models;Security;Convolutional neural networks;Fake news;Western blots;synthetically generated images;image forensics;source attribution;scientific integrity},
  doi={10.1109/WIFS61860.2024.10810680}}
```
### Contact

If you find any issues or bugs, or want to discuss the methods and possible improvements for detecting AI-generated scientific images, please do not hesitate to open an issue.

