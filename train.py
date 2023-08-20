import mne
from constant import experiment_run
from CSPPairwise import CSPPairwise
from data import EegbciData
from itertools import combinations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def train(subject, experiment) :
    data = EegbciData(subject)
    x_train, x_test, y_train, y_test = data.get_exp_data(experiment)
    pair_val = combinations(list(experiment_run[experiment]["event_id"].values()), 2)
    csppairwise = CSPPairwise(pair_val, verbose="ERROR")
    rfc = RandomForestClassifier(n_estimators=150)

    pipeline = Pipeline([
        ('csp', csppairwise),
        ('classifier', rfc)
    ])
    scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring="accuracy")
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))