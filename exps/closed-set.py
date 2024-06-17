"""
Closed-set Experiment Scenario.

This scenario investigates whether the artifacts from each generator can be dis-
tinguished when the classifier is trained using all data sources.
This is the simplest scenario among all experiments.

Authors: Joao Phillipe Cardenuto.
Date: April, 2024
"""
# Models
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
# Utils
from sklearn.preprocessing import label_binarize
import concurrent.futures
import argparse
from  exputils import cross_val_split, cross_domain_split
from exputils import read_features
import numpy as np
import pandas as pd
from tqdm import tqdm
# Set seed
SEED=0
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

def fit_predict(X_train, y_train, X_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_probas = classifier.predict_proba(X_test)
    return y_pred, y_probas

def calculate_auc(pred_probas, y_true):
    """
    Encode the labels to accommodate to the type of classification task (multi or binary)
    and calculate the AUC.

    If the task is multi-class, the AUC is calculated using the One-Vs-Rest strategy,
    using the micro average.
    """
    if len(np.unique(y_true)) > 2: # multi class
        y_true = label_binarize(y_true, classes=np.unique(y_true))
        ovr_auc = roc_auc_score(y_true, pred_probas, average="micro", multi_class="ovr")    
        return ovr_auc
    else: # binary
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        return roc_auc_score(y_true, pred_probas[:, 1])

def calculate_metrics(y_true, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred)
    return bacc

def create_classifier(classifier):
    if classifier.__class__.__name__ == "SVC":
        return SVC(random_state=SEED, probability=True)
    elif classifier.__class__.__name__ == "XGBClassifier":
        return XGBClassifier(random_state=SEED)
    elif classifier.__class__.__name__ == "RandomForestClassifier":
        return RandomForestClassifier(random_state=SEED)
    else:
        raise ValueError("Invalid classifier")

def train_classifier_cv(split1, split2, filter, classifier):
    # Prepare data.
    x_split1 = [ np.array(f) for f in split1[filter].values]
    y_split1 = split1["type"].values
    x_split2 = [ np.array(f) for f in split2[filter].values]
    y_split2 = split2["type"].values
    le = LabelEncoder()
    le.fit(y_split1)
    le_y_split1, le_y_split2 = le.transform(y_split1), le.transform(y_split2)

    # Train classifiers.
    split1_classifier = create_classifier(classifier)
    pred_2, proba_2= fit_predict(x_split1, le_y_split1, x_split2, split1_classifier)
    split2_classifier = create_classifier(classifier)
    pred_1, proba_1 = fit_predict(x_split2, le_y_split2, x_split1, split2_classifier)

    # Calculate metrics for each split.
    # train split1; test split2
    bcc_2 = calculate_metrics(le_y_split2, pred_2)
    auc_2 = calculate_auc(proba_2, y_split2)

    # train split2; test split1
    bcc_1 = calculate_metrics(le_y_split1, pred_1)
    auc_1 = calculate_auc(proba_1, y_split1)

    scores = {}

    scores["bcc_mean"] = np.mean([bcc_1, bcc_2])
    scores["auc_mean"] = np.mean([auc_1, auc_2])
    scores["bcc_std"] = np.std([bcc_1, bcc_2])
    scores["auc_std"] = np.std([auc_1, auc_2])

    return scores

def parallel_training(split1, split2, classifier):
    """
    Run the experiment for each residuum extraction (filter) in parallel.
    """
    results = pd.DataFrame(columns=["classifier", "filter","bcc_mean", "bcc_std",
                                     "auc_mean", "auc_std"])

    def train_filter(filter):
        scores = train_classifier_cv(split1.copy(), split2.copy(), filter, classifier)
        return [classifier.__class__.__name__, filter, scores["bcc_mean"],
                scores["bcc_std"], scores["auc_mean"], scores["auc_std"]]

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


# Parse arg
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
# Classifier
rf = RandomForestClassifier(random_state=SEED)
xgboost = XGBClassifier(random_state=SEED)
svm = SVC(random_state=SEED)

# Run experiments.
all_results = pd.DataFrame(columns=["feat", "classifier", "filter",    
                                    "bcc_mean", "bcc_std",
                                    "auc_mean", "auc_std"])
for feat_name, feat in tqdm(FEATS.items()):
    for classifier in [rf, xgboost, svm]:
        print(f"Processing {feat_name} {classifier.__class__.__name__}")
        split1, split2 = split_func(feat, SEED=SEED)
        result = parallel_training(split1, split2, classifier)
        result["feat"] = feat_name
        all_results = pd.concat([all_results, result])

all_results.to_csv(f"closed_set_split_{split}.csv")