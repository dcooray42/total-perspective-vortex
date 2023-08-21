import mne
import numpy as np
import pickle
from constant import tasks_labels
from CSPPairwise import CSPPairwise
from data import EegbciData
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def train(subject, task) :
    mne.set_log_level("CRITICAL")
    data = EegbciData(subject)
    x_train, x_test, y_train, y_test = data.get_exp_data(task)
    pair_val = combinations(tasks_labels[task], 2)
    csp_pairwise = CSPPairwise(pair_val)
    rfc = RandomForestClassifier()

    pipeline = Pipeline([
        ("csp", csp_pairwise),
        ("classifier", rfc)
    ])
    scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring="accuracy")
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))
    with open("models.pkl", "wb") as f:
        pickle.dump({
            "random_state" : data.rs,
            "pipeline" : pipeline 
        }, f)