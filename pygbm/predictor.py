import numpy as np
from numba import njit, from_dtype, prange
import numba


PREDICTOR_RECORD_DTYPE = np.dtype([
    ('is_leaf', np.uint8),
    ('value', np.float32),
    ('count', np.uint32),
    ('feature_idx', np.uint32),
    ('bin_threshold', np.uint8),
    ('threshold', np.float32),
    ('left', np.uint32),
    ('right', np.uint32),
    ('gain', np.float32),
    ('depth', np.uint32),
    # TODO: shrinkage in leaf for feature importance error bar?
])
PREDICTOR_NUMBA_TYPE = from_dtype(PREDICTOR_RECORD_DTYPE)[::1]


class TreePredictor:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_n_leaf_nodes(self):
        return int(self.nodes['is_leaf'].sum())

    def predict_binned(self, binned_data, out=None):
        if out is None:
            out = np.empty(binned_data.shape[0], dtype=np.float32)
        _predict_binned(self.nodes, binned_data, out)
        return out

    def predict(self, X):
        # TODO: introspect X to dispatch to numerical or categorical data
        # (dense or sparse) on a feature by feature basis.
        out = np.empty(X.shape[0], dtype=np.float32)
        _predict_from_numeric_data(self.nodes, X, out)
        return out


@njit
def _predict_one_binned(nodes, binned_data):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['value']
        if binned_data[node['feature_idx']] <= node['bin_threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


@njit
def _get_threads_chunks(total_size):
    # Divide [0, total_size - 1] into n_threads contiguous regions, and
    # returns the starts and ends of each region. Used to simulate a 'static'
    # scheduling.
    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, total_size // n_threads, dtype=np.int32)
    sizes[:total_size % n_threads] += 1
    starts = np.zeros(n_threads, dtype=np.int32)
    starts[1:] = np.cumsum(sizes[:-1])
    ends = starts + sizes
    return starts, ends, n_threads


@njit(parallel=True)
def _predict_binned(nodes, binned_data, out):
    # for i in prange(binned_data.shape[0]):
    #     out[i] = _predict_one_binned(nodes, binned_data[i])
    starts, ends, n_threads = _get_threads_chunks(binned_data.shape[0])
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            out[i] = _predict_one_binned(nodes, binned_data[i])


@njit
def _predict_one_from_numeric_data(nodes, numeric_data):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['value']
        if numeric_data[node['feature_idx']] <= node['threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


@njit(parallel=True)
def _predict_from_numeric_data(nodes, numeric_data, out):
    # for i in prange(numeric_data.shape[0]):
    #     out[i] = _predict_one_from_numeric_data(nodes, numeric_data[i])
    starts, ends, n_threads = _get_threads_chunks(numeric_data.shape[0])
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            out[i] = _predict_one_from_numeric_data(nodes, numeric_data[i])
