import numpy as np
from mne.decoding import CSP
import matplotlib.pyplot as plt
from CSP import CommonSpatialPatterns
import mne

mne.set_log_level("CRITICAL")
np.random.seed(42)
n_samples = 200
n_channels = 64
n_timepoints = 1000

class_a_data = np.random.randn(n_samples, n_channels, n_timepoints)
class_b_data = np.random.randint(2, size=(n_samples,))

your_csp = CommonSpatialPatterns()
your_csp.fit(class_a_data, class_b_data)
your_transformed_data = your_csp.transform(class_a_data)

mne_csp = CSP()
mne_csp.fit(class_a_data, class_b_data)
mne_transformed_data = mne_csp.transform(class_a_data)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(your_transformed_data[:, 0], your_transformed_data[:, 1], c='blue', label='Your CSP')
plt.title('Transformed Data (Your CSP)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(mne_transformed_data[:, 0], mne_transformed_data[:, 1], c='orange', label="MNE's CSP")
plt.title("Transformed Data (MNE's CSP)")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.tight_layout()
plt.show()