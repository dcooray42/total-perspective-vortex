from constant import experiment_run
from mne import concatenate_epochs, Epochs, events_from_annotations, pick_types
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.io import read_raw_edf, RawEDF
from random import randint
from sklearn.model_selection import train_test_split
from sys import maxsize

class EegbciData() :
    def __init__(self, subject : int, random_state : int =  None) :
        if not isinstance(subject, int) :
            raise ValueError("Subject is not an int.")
        self.subject = subject
        self.rs = random_state if random_state else randint(0, maxsize)
        raw_fnames = eegbci.load_data(self.subject, range(1, 15))
        self.data_list = []
        montage = make_standard_montage("standard_1005")
        for index, f in enumerate(raw_fnames) :
            data = read_raw_edf(f, preload=True)
            data.set_montage(montage)
            data.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
            self.data_list.append(self._normalize_epochs(data, index))
    
    def _normalize_epochs(self, data : RawEDF, index : int) :
        subepoch_duration = 2
        subepochs = []
        all_subepochs = []
        epochs_train = None
        picks = pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    
        if index <= 2 :
            for start_time in range(0, int(data.times[-1]), subepoch_duration):
                end_time = start_time + subepoch_duration
                subraw = data.copy().crop(tmin=start_time, tmax=end_time)
                subepochs.append(subraw)
    
            for subepoch in subepochs:
                events, _ = events_from_annotations(subepoch, event_id=experiment_run[index]["events"])
                subepoch_epochs = Epochs(subepoch,
                                         events,
                                         event_id=experiment_run[index]["event_id"],
                                         tmin=0,
                                         tmax=subepoch_duration,
                                         proj=True,
                                         picks=picks,
                                         baseline=None,
                                         reject=None,
                                         preload=True,
                                         reject_by_annotation=True,
                                         verbose=True)
                all_subepochs.append(subepoch_epochs)
            concat_epochs = concatenate_epochs(all_subepochs)
            epochs_train = concat_epochs.copy().crop(tmin=1.0, tmax=2.0)
        else :
            tmin, tmax = -1.0, 4.0
            events, _ = events_from_annotations(data, event_id=experiment_run[index]["events"])
            all_epochs = Epochs(data,
                                events,
                                event_id=experiment_run[index]["event_id"],
                                tmin=tmin,
                                tmax=tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                reject=None,
                                preload=True,
                                reject_by_annotation=True,
                                verbose=True)
            epochs_train = all_epochs.copy().crop(tmin=1.0, tmax=2.0)
        labels = epochs_train.events[:, -1]
        return train_test_split(epochs_train.get_data(), labels, test_size=0.2, random_state=self.rs)