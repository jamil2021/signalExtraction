# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Load CSV file with new line delimiter
# values = np.loadtxt('data_out.csv', skiprows=1)

# # Perform PCA
# pca = PCA(n_components=1)
# pca.fit(values.reshape(-1, 1))
# values_pca = pca.transform(values.reshape(-1, 1))

# # Perform SVD
# U, S, VT = np.linalg.svd(values.reshape(-1, 1), full_matrices=False)

# # Perform FFT
# fft_result = np.fft.fft(values)

# # Plot the results
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot PCA
# axs[0].plot(values_pca, marker='o')
# axs[0].set_title('PCA')
# axs[0].set_xlabel('Time (minutes)')
# axs[0].set_ylabel('PCA Components')

# # Plot SVD
# axs[1].plot(S, marker='o')
# axs[1].set_title('SVD')
# axs[1].set_xlabel('Singular Value Index')
# axs[1].set_ylabel('Singular Values')

# # Plot FFT
# axs[2].plot(np.abs(fft_result), marker='o')
# axs[2].set_title('Magnitude Spectrum (FFT)')
# axs[2].set_xlabel('Frequency')
# axs[2].set_ylabel('Magnitude')

# plt.tight_layout()
# plt.show()

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Your data
X = np.loadtxt('data_out.csv', skiprows=1)

# Reshape the data to have one feature per row
X = X.reshape(-1, 1)

# Initialize PCA
pca = PCA(n_components=1)

# Fit PCA to the data
pca.fit(X)

# Transform the data to the lower-dimensional space
X_pca = pca.transform(X)

# Plot the original data and the principal components
plt.figure(figsize=(10, 6))
plt.scatter(X, np.zeros_like(X), label='Original Data')
plt.scatter(X_pca, np.zeros_like(X_pca), label='PCA Components', marker='x')
plt.xlabel('Data')
plt.ylabel('PCA Components')
plt.title('PCA on Data')
plt.legend()
plt.show()
