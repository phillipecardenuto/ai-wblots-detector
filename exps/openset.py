
"""
Open-set scenario: In this scenario, the classifier is trained using data from
real sources and tested on data from synthetic sources.

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""
# Models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
# Utils
import concurrent.futures
import argparse
from exputils import cross_val_split, cross_domain_split
from exputils import read_features
import numpy as np
import pandas as pd
from tqdm import tqdm

# set seed
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

def calculate_score(y_true, y_prob, metric_func):
    """
    Use the roc curve to find the best threshold for the classifier.
    y_true : array-like of shape (n_samples,) with the true labels
    y_prob : array-like of shape (n_samples,) with the probability of the positive class
    metric : function to evaluate the threshold

    return the score and best threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    scores = [metric_func(y_true, np.where(y_prob >= threshold, 1, -1)) 
              for threshold in thresholds]
    return max(scores), thresholds[np.argmax(scores)]

def train_one_class_classifier(X_real_train, X_test, y_test, classifier):

    # Train step
    classifier.fit(X_real_train)

    # Pred step
    if classifier.__class__.__name__ == "IsolationForest":
         y_proba = classifier.decision_function(X_test)
    else:
        y_proba = classifier.score_samples(X_test)
    y_proba = np.nan_to_num(y_proba)

    auc = roc_auc_score(y_test, y_proba)

    # best_threshold = find_best_threshold(classifier, X_test, y_test)
    auc = roc_auc_score(y_test, y_proba)
    bcc, bcc_th = calculate_score(y_test, y_proba, balanced_accuracy_score)

    return bcc, auc

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
    scores = {"bcc":[], "auc":[]}
    
    x_split1 = [ np.array(f) for f in split1[filter].values]
    y_split1 = split1["type"].values
    x_split2 = [ np.array(f) for f in split2[filter].values]
    y_split2 = split2["type"].values

    y_split1 = np.where(y_split1 == "real", 1, -1)
    y_split2 = np.where(y_split2 == "real", 1, -1)

    # Calculate metrics for each split.
    # train split1; test split2
    gen_model = get_new_classifier(classifier)
    real_split1 = split1[split1["type"] == "real"]
    real_split1 = [ np.array(f) for f in real_split1[filter].values]
    real_split1 = np.array(real_split1)
    bcc, auc = train_one_class_classifier(real_split1, x_split2, y_split2, gen_model)
    scores["bcc"].append(bcc); scores["auc"].append(auc)
    
    # train split2; test split1
    gen_model = get_new_classifier(classifier)
    real_split2 = split2[split2["type"] == "real"]
    real_split2 = [ np.array(f) for f in real_split2[filter].values]
    bcc, auc = train_one_class_classifier(real_split2, x_split1, y_split1, gen_model)
    scores["bcc"].append(bcc); scores["auc"].append(auc)

    scores["bcc_mean"] = np.mean(scores["bcc"])
    scores["auc_mean"] = np.mean(scores["auc"])
    scores["bcc_std"] = np.std(scores["bcc"])
    scores["auc_std"] = np.std(scores["auc"])

    return scores

def parallel_training(split1, split2, classifier):
    """
    Check the performance of each residuum extraction (filter) in parallel.
    """
    results = pd.DataFrame(columns=["classifier", "filter", "bcc_mean", "bcc_std",
                                    "auc_mean", "auc_std"])

    def train_filter(filter):
        scores = train_classifier_cv(split1.copy(), split2.copy(), filter, classifier)
        return [classifier.__class__.__name__, filter, 
                scores["bcc_mean"], scores["bcc_std"],
                scores["auc_mean"], scores["auc_std"]]

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

# Classifiers
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

all_results.to_csv(f"openset_split_{split}.csv")