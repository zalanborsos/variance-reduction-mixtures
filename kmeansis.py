import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.k_means_ import _labels_inertia

from sklearn.utils.extmath import row_norms
import time


class KmeansIS(MiniBatchKMeans):
    def __init__(self, n_clusters=8, sampler=None, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01, step_size_init=1):

        super(KmeansIS, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels, random_state=random_state, tol=tol,
            max_no_improvement=max_no_improvement, init_size=init_size, n_init=n_init,
            reassignment_ratio=reassignment_ratio)

        self.sampler = sampler
        self.step_size_init = step_size_init

    def fit(self, X):

        self.log = {'loss': []}
        n_samples, n_features = X.shape
        x_squared_norms = row_norms(X, squared=True)
        distances = np.zeros(self.batch_size, dtype=X.dtype)

        init_size = self.init_size
        self.init_size_ = init_size

        self.cluster_centers_ = self.init
        self.counts_ = np.zeros(self.n_clusters, dtype=np.float32)

        for iteration_idx in range(self.max_iter):
            if iteration_idx % 10 == 0:
                self.log['loss'].append(np.sum((X - self.cluster_centers_[self.predict(X)]) ** 2))

            if self.sampler is None:
                minibatch_indices = self.random_state.randint(
                    0, n_samples, self.batch_size)
                weights = np.ones(self.batch_size)
            else:
                minibatch_indices, weights = self.sampler.sample(self.batch_size)

            X_weighted = X[minibatch_indices] * weights[:, np.newaxis]

            batch_inertia, loss = self._mini_batch_step(
                X[minibatch_indices], x_squared_norms[minibatch_indices], X_weighted, weights,
                self.cluster_centers_, self.counts_, distances=distances)

            if self.sampler is not None:
                self.sampler.update(loss)

        return self

    def _mini_batch_step(self, X, x_squared_norms, X_weighted, weights, centers, counts,
                         distances):

        nearest_center, inertia = _labels_inertia(X, np.ones(X.shape[0]), x_squared_norms, centers,
                                                  distances=distances)
        loss = 4 * np.sum((centers[nearest_center] - X) ** 2, axis=1)

        k = centers.shape[0]
        for center_idx in range(k):

            center_mask = nearest_center == center_idx
            count = (center_mask * weights).sum()

            if count > 0:
                centers[center_idx] *= counts[center_idx]
                centers[center_idx] += np.sum(X_weighted[center_mask], axis=0)
                counts[center_idx] += count
                centers[center_idx] /= counts[center_idx]

        return inertia, loss
