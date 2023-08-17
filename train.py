import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train(subject, experiment) :
    raw_data = mne.io.read_raw_edf(f"/Users/dimitricooray/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{str(subject).zfill(3)}/S{str(subject).zfill(3)}R{str(experiment).zfill(2)}.edf", preload=True)
    raw_data.rename_channels(lambda x: x.strip(".").upper().replace("FP", "Fp").replace("Z", "z"))
    raw_data.set_montage("standard_1020")
    raw_data.set_eeg_reference('average', projection=True)
    raw_data.filter(l_freq=1, h_freq=50)

    events, _ = mne.events_from_annotations(raw_data)
    tmin, tmax = -1.0, 2.0
    epochs = mne.Epochs(raw_data,
                        events,
                        event_id=None,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        reject=None,
                        preload=True,
                        on_missing="error",
                        reject_by_annotation=True,
                        verbose=True)
    
    freq_bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
    features = np.empty((len(epochs), len(freq_bands)))
    for i, epoch in enumerate(epochs.get_data()):
        for j, (fmin, fmax) in enumerate(freq_bands):
            time_indices = (epochs.times >= tmin) & (epochs.times <= tmax)
            power = np.mean(np.abs(epoch[:, :, time_indices]) ** 2)
            features[i, j] = power
    
    csp = CSP(n_components=6)  # You can adjust the number of components
    classifier = SVC(kernel='linear')

    pipeline = Pipeline([
        ('csp', csp),
        ('classifier', classifier)
    ])

    # Extract labels from event_ids
    labels = epochs.events[:, -1]
    print(labels)

    # Fit and evaluate the pipeline using cross-validation
    scores = cross_val_score(pipeline, features, labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))