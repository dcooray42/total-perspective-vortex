import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def train(subject, experiment) :
    raw_data = mne.io.read_raw_edf(f"/Users/dimitricooray/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{str(subject).zfill(3)}/S{str(subject).zfill(3)}R{str(experiment).zfill(2)}.edf", preload=True)
#    raw_data.rename_channels(lambda x: x.strip(".").upper().replace("FP", "Fp").replace("Z", "z"))
    mne.datasets.eegbci.standardize(raw_data)  # set channel names
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_data.set_montage(montage)
#    raw_data.set_montage("standard_1020")
#    raw_data.set_eeg_reference('average', projection=True)
#    raw_data.filter(1, 50)
    raw_data.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, _ = mne.events_from_annotations(raw_data, event_id=dict(T0=1, T1=2, T2=3))
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    tmin, tmax = -1.0, 4.0
    subepoch_duration = 2  # Duration of each subepoch in seconds
    subepochs = []
    
    for start_time in range(0, int(raw_data.times[-1]), subepoch_duration):
        end_time = start_time + subepoch_duration
        subraw = raw_data.copy().crop(tmin=start_time, tmax=end_time)
        subepochs.append(subraw)

    for subepoch in subepochs:
        events, _ = mne.events_from_annotations(subepoch, event_id=dict(T0=1, T1=2, T2=3))
        subepoch_epochs = mne.Epochs(subepoch, events, event_id=dict(rest=1),
                                     tmin=0, tmax=subepoch_duration, proj=True,
                                     picks=picks, baseline=None, reject=None,
                                     preload=True, reject_by_annotation=True,
                                     verbose=True)
#    epochs = mne.Epochs(raw_data,
#                        events,
#                        event_id=dict(rest=1),
#                        tmin=tmin,
#                        tmax=tmax,
#                        proj=True,
#                        picks=picks,
#                        baseline=None,
#                        reject=None,
#                        preload=True,
#                        reject_by_annotation=True,
#                        verbose=True)
    
    epochs_train = subepoch_epochs.copy().crop(tmin=1.0, tmax=2.0)
    
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)  # You can adjust the number of components
#    classifier = SVC(kernel='linear')
#    lda = LinearDiscriminantAnalysis()
#
#    pipeline = Pipeline([
#        ('csp', csp),
#        ('classifier', lda)
#    ])
#
#    # Extract labels from event_ids
    labels = subepoch_epochs.events[:, -1]
    print(labels)
    csp.fit(subepoch_epochs.get_data(), labels)
    csp.transform(subepoch_epochs.get_data())
#
#    # Fit and evaluate the pipeline using cross-validation
#    scores = cross_val_score(pipeline, epochs_train.get_data(), labels, cv=5)
#    print("Cross-validation scores:", scores)
#    print("Mean accuracy:", np.mean(scores))