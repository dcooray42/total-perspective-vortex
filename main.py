import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
import numpy as np

raw_data = mne.io.read_raw_edf("/Users/dimitricooray/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R03.edf", preload=True)
standard_montage = mne.channels.make_standard_montage("standard_1020")
raw_data.rename_channels(lambda x: x.strip(".").upper().replace("FP", "Fp").replace("Z", "z"))
print(raw_data.ch_names)
print(raw_data)
print(raw_data.info)
data_ch_names = set(raw_data.ch_names)
montage_ch_names = set(standard_montage.ch_names)

# Find the missing channels in the montage
missing_channels = data_ch_names - montage_ch_names
print("Missing channels:", missing_channels)
raw_data.set_montage("standard_1020")
raw_data.plot_sensors(show_names=True)
plt.show()
raw_data.plot(n_channels=64, color="red", remove_dc=False)
plt.show()
raw_data.set_eeg_reference('average', projection=True)
raw_data.filter(l_freq=1, h_freq=50)
raw_data.plot(n_channels=64, color="red", remove_dc=False)
plt.show()
raw_data.compute_psd(fmax=80).plot(picks="data", exclude="bads")
plt.show()
raw_data.compute_psd(fmax=80).plot(average=True, picks="data", exclude="bads")
plt.show()

selected_features = ["AFz", "Fz", "FC1", "FC2", "FC5", "FC6", "C3", "C4", "CP5", "CP6", "P1", "P2", "POz", "Iz"]

n_components=14

raw_data.pick(selected_features)

ica = ICA(n_components=n_components, random_state=97)  # You can adjust the number of components
ica.fit(raw_data)

mixing_matrix = ica.unmixing_matrix_  # Get the mixing matrix (projection matrix)
explained_variances = np.sum(mixing_matrix ** 2, axis=1) / np.sum(mixing_matrix ** 2)

# Plot explained variance
plt.figure()
plt.plot(range(1, n_components + 1), explained_variances, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.show()

raw_corrected = ica.apply(raw_data, exclude=[], n_pca_components=4)

raw_corrected.compute_psd(fmax=80).plot(picks="data", exclude="bads")
plt.show()
raw_corrected.compute_psd(fmax=80).plot(average=True, picks="data", exclude="bads")
plt.show()

raw_corrected.plot(n_channels=64, color="red", remove_dc=False)
plt.show()

print("")
print(f"raw data shape = {raw_data._data.shape}\nraw corrected shape = {raw_corrected._data.shape}")
print("")

events, _ = mne.events_from_annotations(raw_corrected)
#raw_data.apply_baseline(baseline=(None, 0), verbose=True)
tmin, tmax = -1.0, 2.0
epochs = mne.Epochs(raw_corrected,
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