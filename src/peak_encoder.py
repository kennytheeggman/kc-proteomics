import h5py
import numpy as np

# need to do it on torch instead? and make into a function that does it all at once

filename = "/home/kc/workspace/datasets/IVE_v2/IVE_v2_train.h5"

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())  # keys are ['charges', 'mapping', 'peptides', 'premzs', 'spectra']
    spectra = f['spectra'][:10]

# one spectrum at a time maybe
spectrum = spectra[0]
len_full = len(spectrum)

# initialize variables
m_max = 1000
m_min = 10E-4
n_peaks = 60
k = int(1/m_min)
len_b = m_max + k

# initialize arrays
raw_encode = np.zeros((2, n_peaks))
fourier_encode = np.zeros((2*len_b, n_peaks))
b = (np.concatenate((np.arange(m_max, 0, -1), m_min*np.arange(k, 0, -1)))).reshape(-1, 1)

# fill in raw encoding
assert len_full%2 == 0
len_half = int(len_full/2)
mzs = np.array(spectrum[:len_half])
intensities = np.array(spectrum[len_half:])
n_sort = min(len_half, n_peaks)
n_indices = np.argsort(intensities)[-n_sort:]  # indices of top n_sort intensities
raw_encode[0, :n_sort] = mzs[n_indices]
raw_encode[1, :n_sort] = intensities[n_indices]

# fill in fourier encoding
sine_terms = np.sin(raw_encode[0]*b*2*np.pi)
cosine_terms = np.cos(raw_encode[0]*b*2*np.pi)
fourier_encode[::2, :] = sine_terms
fourier_encode[1::2, :] = cosine_terms
