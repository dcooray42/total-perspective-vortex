import mne
import numpy as np
from constant import tasks
from CSPPairwise import CSPPairwise
from data import EegbciData
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def score() :
    mne.set_log_level("CRITICAL")
    mean_exp_accuracy = [[] for _ in range(0, 6)]
    for subject in range(1, 110) :
        data = EegbciData(subject)
        x_train, x_test, y_train, y_test = data.get_exp_data()
        pair_val = combinations(range(1, 11), 2)
        csp_pairwise = CSPPairwise(pair_val)
        rfc = RandomForestClassifier(random_state=42)

        pipeline = Pipeline([
            ("csp", csp_pairwise),
            ("classifier", rfc)
        ])
        pipeline.fit(x_train, y_train)
        for exp in range(0, 6) :
            shape_epoch = x_test[exp].shape
            x_test_exp = np.empty((0, shape_epoch[1], shape_epoch[2]))
            y_test_exp = np.array([])
            if exp < 2 :
                x_test_exp = np.concatenate((x_test_exp,
                                             x_test[exp]))
                y_test_exp = np.concatenate((y_test_exp,
                                             y_test[exp]))
            else :
                for exp_run in tasks[exp - 1] :
                    x_test_exp = np.concatenate((x_test_exp,
                                                 x_test[exp_run]))
                    y_test_exp = np.concatenate((y_test_exp,
                                                 y_test[exp_run]))
            score = pipeline.score(x_test_exp, y_test_exp)
            mean_exp_accuracy[exp].append(score)
            print(f"Experiment {exp}: Subject {str(subject).zfill(3)}: accuracy = {round(score, 4)}")
    print("Mean accuracy of the six different experiments for all 109 subjects:")
    for exp in range(0, 6) :
        mean_exp_accuracy[exp] = np.mean(mean_exp_accuracy[exp])
        print(f"experiment {exp}: accuracy = {round(mean_exp_accuracy[exp], 4)}")
    print(f"Accuracy: {round(np.mean(mean_exp_accuracy), 4)}")