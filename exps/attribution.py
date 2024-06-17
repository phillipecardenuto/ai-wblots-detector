
"""
One-vs-Rest Source Attribution

This scenario investigates whether the artifacts from each generator can be dis-
tinguished when the classifier is trained using only one known
data source, which can be real or synthetic. 

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""

# Models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
# Utils
import concurrent.futures
import argparse
from exputils import cross_val_split, cross_domain_split
from exputils import read_features
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from tqdm import tqdm
import numpy as np
import pandas as pd

# Set Seed
SEED = 0
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on split type")
    parser.add_argument("--split", type=str, required=True, help="Options"+\
                        " are: xval, xdomain")
    args = parser.parse_args()
    split = args.split
    if split == "xval":
        split_func = cross_val_split
    elif split == "xdomain":
        split_func = cross_domain_split
    else:
        raise ValueError("Invalid split type")
    return split, split_func

def fit_transform_one_class_classifier(X_train, X_test, classifier):
    classifier.fit(X_train)
    if classifier.__class__.__name__ == "IsolationForest":
        y_pred = classifier.decision_function(X_test)
    else:
        y_pred = classifier.score_samples(X_test)
    y_pred = np.nan_to_num(y_pred)
    return y_pred

def calculate_auc(pred_split):
    y_true = label_binarize(pred_split["y_true"], classes=pred_split.columns[1:])
    y_pred = pred_split.drop(columns="y_true").values
    ovr_auc = roc_auc_score(y_true, y_pred, average="micro", multi_class="ovr")
    return ovr_auc

def calculate_attribution_metrics(y_pred, y_true):
    bcc = balanced_accuracy_score(y_true, y_pred)
    return bcc

def get_new_classifier(classifier):
    """Return a non-trained classifier instance, based on the input classifier."""
    if classifier.__class__.__name__ == "IsolationForest":
        return IsolationForest(random_state=SEED)
    elif classifier.__class__.__name__ == "PCA":
        return PCA(n_components=0.95, random_state=SEED)
    elif classifier.__class__.__name__ == "OneClassSVM":
        return OneClassSVM()
    else:
        raise ValueError("Invalid classifier")

def train_classifier_cv(split1, split2, filter, classifier):
    # Prepare data.
    x_split1 = [ np.array(f) for f in split1[filter].values]
    y_split1 = split1["type"].values
    x_split2 = [ np.array(f) for f in split2[filter].values]
    y_split2 = split2["type"].values
    pred_split1 = pd.DataFrame(columns=["y_true"])
    pred_split2 = pd.DataFrame(columns=["y_true"])
    pred_split1["y_true"] = y_split1
    pred_split2["y_true"] = y_split2

    # Train classifiers.
    for generator in split1.type.unique():
        gen_model = get_new_classifier(classifier)
        # fit on split2 and predict on split1
        fit_x = [ np.array(f) for f in split2[split2["type"]==generator][filter].values]
        pred_split1[generator] = fit_transform_one_class_classifier(fit_x, x_split1, gen_model)

        gen_model = get_new_classifier(classifier)
        # fit on split1 and predict on split2
        fit_x = [ np.array(f) for f in split1[split1["type"]==generator][filter].values]
        pred_split2[generator] = fit_transform_one_class_classifier(fit_x, x_split2, gen_model)

    # Calculate scores.
    # train split1; test split2
    auc_1 = calculate_auc(pred_split1)
    pred_split1["y_pred"] = pred_split1.drop(columns="y_true").idxmax(axis=1)
    bcc_1 = calculate_attribution_metrics(pred_split1["y_pred"], pred_split1["y_true"])
    # train split2; test split1
    auc_2 = calculate_auc(pred_split2)
    pred_split2["y_pred"] = pred_split2.drop(columns="y_true").idxmax(axis=1)
    bcc_2 = calculate_attribution_metrics(pred_split2["y_pred"], pred_split2["y_true"])

    scores = {}

    scores["bcc_mean"] = np.mean([bcc_1, bcc_2])
    scores["auc_mean"] = np.mean([auc_1, auc_2])
    scores["bcc_std"] = np.std([bcc_1, bcc_2])
    scores["auc_std"] = np.std([auc_1, auc_2])

    return scores

def parallel_training(split1, split2, classifier):
    """
    Check the performance of each residuum extraction (filter) in parallel.
    """

    results = pd.DataFrame(columns=["classifier", "filter",
                                    "bcc_mean", "bcc_std",
                                    "auc_mean", "auc_std"])
    def train_filter(filter):
        scores = train_classifier_cv(split1.copy(), split2.copy(), filter, classifier)
        return [classifier.__class__.__name__, filter, scores["f1_mean"],
                scores["f1_std"], scores["bcc_mean"], scores["bcc_std"], scores["auc_mean"],
                scores["auc_std"]]
    local_filters = [f for f in FILTERS if f in split1.columns]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_filter = {executor.submit(train_filter, filter): filter for filter in local_filters}
        for future in concurrent.futures.as_completed(future_to_filter):
            try:
                result = future.result()
                results.loc[len(results)] = result
            except Exception as exc:
                print(f'Filter {future_to_filter[future]} generated an exception: {exc}')
    return results

# Parse args.
split, split_func = parse_args()
# Read features.
FEATS = read_features()
FILTERS = ["gaussian",
        "access",
        "matthias",
        "mean",
        "cross",
        "nlmeans",
        "pmap",
        "none",
]
# Classifiers.
iforest = IsolationForest(random_state=SEED)
svm = OneClassSVM()
pca = PCA(n_components=0.95, random_state=SEED)
classifiers = [iforest, svm, pca]

all_results = pd.DataFrame(columns=["feat", "classifier", "filter",
                                    "bcc_mean", "bcc_std",
                                    "auc_mean", "auc_std"])
for feat_name, feat in tqdm(FEATS.items()):
    for classifier in  classifiers:
        print(f"Processing {feat_name} {classifier.__class__.__name__}")
        split1, split2 = split_func(feat, SEED=SEED)
        result = parallel_training(split1, split2, classifier)
        result["feat"] = feat_name
        all_results = pd.concat([all_results, result])

all_results.to_csv(f"attribution_split_{split}.csv")