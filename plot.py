import mne
import matplotlib.pyplot as plt

def plot(subject, experiment) :
    raw_data = mne.io.read_raw_edf(f"/Users/dimitricooray/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{str(subject).zfill(3)}/S{str(subject).zfill(3)}R{str(experiment).zfill(2)}.edf", preload=True)
#    standard_montage = mne.channels.make_standard_montage("standard_1020")
#    raw_data.rename_channels(lambda x: x.strip(".").upper().replace("FP", "Fp").replace("Z", "z"))
    print(raw_data.ch_names)
    print(raw_data)
    print(raw_data.info)
#    data_ch_names = set(raw_data.ch_names)
#    montage_ch_names = set(standard_montage.ch_names)

    # Find the missing channels in the montage
#    missing_channels = data_ch_names - montage_ch_names
#    print("Missing channels:", missing_channels)
#    raw_data.set_montage("standard_1020")
    mne.datasets.eegbci.standardize(raw_data)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_data.set_montage(montage)
    raw_data.plot_sensors(show_names=True)
    plt.show()
    raw_data.plot(n_channels=64, color="red", remove_dc=False)
    plt.show()
    raw_data.compute_psd(fmax=80).plot(picks="data", exclude="bads")
    plt.show()
    raw_data.compute_psd(fmax=80).plot(average=True, picks="data", exclude="bads")
    plt.show()
    raw_data.set_eeg_reference('average', projection=True)
    raw_data.filter(l_freq=1, h_freq=50)
    raw_data.plot(n_channels=64, color="red", remove_dc=False)
    plt.show()
    raw_data.compute_psd(fmax=80).plot(picks="data", exclude="bads")
    plt.show()
    raw_data.compute_psd(fmax=80).plot(average=True, picks="data", exclude="bads")
    plt.show()

    events, _ = mne.events_from_annotations(raw_data)
    #raw_data.apply_baseline(baseline=(None, 0), verbose=True)
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
    print(epochs)
    raw_data.del_proj()
    print(events)
    print(epochs._data)
    print(epochs._data.shape)