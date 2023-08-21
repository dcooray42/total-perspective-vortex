import mne
import numpy as np
import pickle
from data import EegbciData
from random import randint
from sklearn.metrics import accuracy_score

def predict(subject, task) :
    try :
        with open("models.pkl", "rb") as f :
            pkl_data = pickle.load(f)
        random_state = pkl_data["random_state"]
        pipeline = pkl_data["pipeline"]
    except :
        raise
    mne.set_log_level("CRITICAL")
    data = EegbciData(subject, random_state=random_state)
    x_train, x_test, y_train, y_test = data.get_exp_data(task)
    shape_epoch = x_test.shape
    y_true = np.array([])
    y_predict = np.array([])
    print(y_test.shape)
    print("epoch nb: [prediction] [truth] equal?")
    try :
        epoch = 0
        while True :
            index = randint(0, y_test.shape[0] - 1)
            y_hat = pipeline.predict(x_test[index].reshape(1, shape_epoch[1], shape_epoch[2]))
            y_true_index = y_test[index].reshape(1,)
            y_true = np.concatenate((y_true,
                                     y_true_index))
            y_predict = np.concatenate((y_predict,
                                       y_hat))
            print(f"epoch {str(epoch).zfill(2)}: {y_hat[0]} {y_true_index[0]} {y_hat[0] == y_true_index[0]}")
            epoch += 1
    except (KeyboardInterrupt) :
        print(f"Accuracy: {accuracy_score(y_true, y_predict)}")