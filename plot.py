import mne
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import read_raw_edf

def plot(subject, experiment) :
    raw_fnames = eegbci.load_data(subject, experiment)
    for f in raw_fnames :
        raw_data = read_raw_edf(f, preload=True)
    print(raw_data.ch_names)
    print(raw_data)
    print(raw_data.info)
    fmax = 80 if subject != 88 else 64
    mne.datasets.eegbci.standardize(raw_data)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_data.set_montage(montage)
    raw_data.plot_sensors(show_names=True)
    plt.show()
    raw_data.plot(n_channels=64, color="red", remove_dc=False)
    plt.show()
    raw_data.compute_psd(fmax=fmax).plot(picks="data", exclude="bads")
    plt.show()
    raw_data.compute_psd(fmax=fmax).plot(average=True, picks="data", exclude="bads")
    plt.show()
    raw_data.set_eeg_reference("average", projection=True)
    raw_data.filter(l_freq=7, h_freq=30)
    raw_data.plot(n_channels=64, color="red", remove_dc=False)
    plt.show()
    raw_data.compute_psd(fmax=fmax).plot(picks="data", exclude="bads")
    plt.show()
    raw_data.compute_psd(fmax=fmax).plot(average=True, picks="data", exclude="bads")
    plt.show()

    events, _ = mne.events_from_annotations(raw_data)
    tmin, tmax = -1.0, 4.0
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
    print(epochs)
    raw_data.del_proj()
    print(events)
    print(epochs._data)
    print(epochs._data.shape)