import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_std = self.standardize(X)
        cov_matrix = self.compute_covariance_matrix(X_std)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    def transform(self, X):
        X_std = self.standardize(X)
        return np.dot(X_std, self.components)

    def standardize(self, X):
        return (X - self.mean) / np.std(X, axis=0)

    def compute_covariance_matrix(self, X):
        n_samples = X.shape[0]
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                cov_matrix[i, j] = np.sum((X[:, i] - np.mean(X[:, i])) * (X[:, j] - np.mean(X[:, j]))) / (n_samples - 1)
        return cov_matrix