from time import time

import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from pygbm.grower import TreeGrower
from pygbm.binning import BinMapper
from pygbm.loss import LeastSquares


# TODO: make some random numerical data, bin it, and fit a grower with many
# leaves in a joblib'ed function.

n_samples = int(5e6)

X, y = make_regression(n_samples=n_samples, n_features=5)
bin_mapper = BinMapper()
X_binned = bin_mapper.fit_transform(X)

loss = LeastSquares()
gradients, hessians = loss.init_gradients_and_hessians(
    n_samples, n_trees_per_iteration=1)
y_pred_init = np.zeros_like(y)
loss.update_gradients_and_hessians(gradients, hessians, y, y_pred_init)

grower = TreeGrower(X_binned, gradients, hessians, max_leaf_nodes=None)
grower.grow()

predictor = grower.make_predictor(bin_thresholds=bin_mapper.bin_thresholds_)
X_binned_c = np.ascontiguousarray(X_binned)
print("Compiling predictor code...")
tic = time()
predictor.predict_binned(np.asfortranarray(X_binned[:100]))
predictor.predict_binned(X_binned_c[:100])
predictor.predict(np.asfortranarray(X[:100]))
predictor.predict(X[:100])
toc = time()
print(f"done in {toc - tic:0.3f}s")

data_size = X_binned.nbytes
print("Computing predictions (F-contiguous binned data)...")
tic = time()
scores_binned_f = predictor.predict_binned(X_binned)
toc = time()
duration = toc - tic
print(f"done in {duration:.4f}s ({data_size / duration / 1e9:.3} GB/s)")

print("Computing predictions (C-contiguous binned data)...")
tic = time()
scores_binned_c = predictor.predict_binned(X_binned_c)
toc = time()
duration = toc - tic
print(f"done in {duration:.4f}s ({data_size / duration / 1e9:.3} GB/s)")

assert_allclose(scores_binned_f, scores_binned_c)

data_size = X.nbytes
print("Computing predictions (F-contiguous numerical data)...")
tic = time()
scores_f = predictor.predict(np.asfortranarray(X))
toc = time()
duration = toc - tic
print(f"done in {duration:.4f}s ({data_size / duration / 1e9:.3} GB/s)")

assert_allclose(scores_binned_f, scores_f)

print("Computing predictions (C-contiguous numerical data)...")
data_size = X.nbytes
tic = time()
scores_c = predictor.predict(X)
toc = time()
duration = toc - tic
print(f"done in {duration:.4f}s ({data_size / duration / 1e9:.3} GB/s)")

assert_allclose(scores_binned_f, scores_c)
