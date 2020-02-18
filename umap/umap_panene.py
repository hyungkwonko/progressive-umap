# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function
from warnings import warn
import time
import random

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import numba

import umap.distances as dist

import umap.sparse as sparse

from umap.utils import tau_rand_int, deheap_sort, submatrix, ts, measure_time
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.nndescent import (
    make_nn_descent,
    make_initialisations,
    make_initialized_nnd_search,
    initialise_search,
)
from umap.spectral import spectral_layout

import locale

from pynene import KNNTable
import math

from matplotlib import pyplot as plt
import os
import pathlib

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

@measure_time
@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    result: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rho: array of shape(n_samples)
        The local connectivity adjustment.
    """
    target = np.log2(k) * bandwidth # 2.3192...
    rho = np.zeros(distances.shape[0]) # (1797, 1)
    result = np.zeros(distances.shape[0]) # (1797, 1), distances.shape = (1797, 5)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ## CALCULATE rho[i]
        ith_distances = distances[i] # ith_distances.shape = (1, 5)
        non_zero_dists = ith_distances[ith_distances > 0.0] # select elements > 0.0
        if non_zero_dists.shape[0] >= local_connectivity: # non_zero_dists.shape[0] = number of locally connected dots
            index = int(np.floor(local_connectivity)) # local_connectivity = 1.0
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE: # SMOOTH_K_TOLERANCE = 1e-5
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else: # if index == 0 (0 <= local_connectivity < 1)
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        ## CALCULATE result[i]
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]): # range(1,k) => 1,2,...,k-1
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            # break if it is lower than the threshold
            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE: # fabs: Compute the absolute values element-wise.
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances
                
    return result, rho

@measure_time
def nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=False):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    verbose: bool
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        if callable(metric):
            distance_func = metric
        elif metric in dist.named_distances:
            distance_func = dist.named_distances[metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        if metric in ("cosine", "correlation", "dice", "jaccard"):
            angular = True

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[metric]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds["n_features"] = X.shape[1]
            else:
                raise ValueError("Metric {} not supported for sparse data".format(metric))
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )

            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))
            if verbose:
                print(ts(), "Building RP forest with", str(n_trees), "trees")

            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)

            if verbose:
                print(ts(), "NN descent for", str(n_iters), "iterations")
            knn_indices, knn_dists = metric_nn_descent(
                X.indices,
                X.indptr,
                X.data,
                X.shape[0],
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )
        else:
            metric_nn_descent = make_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            if verbose:
                print(ts(), "Building RP forest with", str(n_trees), "trees")
            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            if verbose:
                print(ts(), "NN descent for", str(n_iters), "iterations")
            knn_indices, knn_dists = metric_nn_descent(
                X,
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )

        if np.any(knn_indices < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )
    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    
    return knn_indices, knn_dists, rp_forest

@measure_time
@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1] # k

    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
            # sum of the vals will be the same as log2(k)*bandwidth

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals

@measure_time
def fuzzy_simplicial_set(X, 
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    verbose=False):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose
        )

    sigmas, rhos = smooth_knn_dist(
        knn_dists, n_neighbors, local_connectivity=local_connectivity
    )
    
    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )
    
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    ) # result.shape = (n, n) adjacency matrix
    result.eliminate_zeros()

    transpose = result.transpose()

    prod_matrix = result.multiply(transpose)

    result = (
        set_op_mix_ratio * (result + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )

    result.eliminate_zeros()
    
    return result


@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return


def reset_local_connectivity(simplicial_set):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.

    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.

    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


def categorical_simplicial_set_intersection(
    simplicial_set, target, unknown_dist=1.0, far_dist=5.0):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from categorical data using categorical distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.

    TODO: optional category cardinality based weighting of distance

    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    fast_intersection(
        simplicial_set.row,
        simplicial_set.col,
        simplicial_set.data,
        target,
        unknown_dist,
        far_dist,
    )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight):

    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        weight,
    )

    return result


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit("f4(f4[:],f4[:])", fastmath=True)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result


@measure_time
@numba.njit(fastmath=True, parallel=True)
def optimize_layout(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding

@measure_time
def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    verbose):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.

    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

    n_components: int
        The dimensionality of the euclidean space into which to embed the data.

    initial_alpha: float
        Initial learning rate for the SGD.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    gamma: float
        Weight to apply to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.

    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    
    optimize_start = ts()
    
    embedding = optimize_layout(
        embedding,
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
        verbose=verbose,
    )
    
    optimize_end = ts()
    print(optimize_end - optimize_start, "OPTIMIZE TIME")

    return embedding

@numba.njit(fastmath=True, parallel=True)
def optimize_layout2(
    head_embedding,
    tail_embedding,
    data,
    head,
    tail,
    eps,
    n_epochs,
    batch_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False):

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_sample[epochs_per_sample > batch_epochs ] = batch_epochs
    epoch_of_next_sample = epochs_per_sample.copy()

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()

    for n in range(batch_epochs):
        alpha = initial_alpha - float(eps) * 0.9 * 100 / float(n_epochs)
        if alpha < 0.1:
            alpha = 0.1
        # alpha = initial_alpha * (1.0 - float(eps) / float(n_epochs))

        if eps >= n_epochs:
            break

        cost = 0
        bit_array = [0] * epochs_per_sample.shape[0] # numba does not support bitarray, we use list instead

        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n: # loss at 0 epoch because of this

                j = head[i] # first index
                k = tail[i] # second index

                current = head_embedding[j] # position of j index in embedded space
                other = tail_embedding[k] # position of k index in embedded space

                dist_squared = rdist(current, other) # if embedding space is 2D (x1-x2)^2 + (y1-y2)^2

                if dist_squared > 0.0: # if they are not the same pts
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                    if bit_array[i] == 0: # if this edge has not been considered, add this to the cost
                        c1 = 1.0 / (1.0 + a * pow(dist_squared, 2 * b))
                        cost -= data[i] * math.log(c1)
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                        if bit_array[i] == 0: # the above edge's negative samples are considered, add this to the cost
                            c2 = a * pow(dist_squared, 2.0 * b) / (1.0 + a * pow(dist_squared, 2.0 * b))
                            cost -= data[i] * gamma * math.log(c2)
                            if p == (n_neg_samples - 1):
                                bit_array[i] = 1 # this edge will not be considered in this epoch anymore
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        if n != 0:
            sum_bit_array = 1e-4 # prevent 0 division
            for l in range(len(bit_array)):
                sum_bit_array += bit_array[l] # calculate the total number of edges considered in this epoch

            if verbose and n % 50 == 0:
                print("\tcompleted ", eps, " / ", n_epochs, "epochs\t, cost/sum_bit_array: ", cost / sum_bit_array,
                    ",\t epochs_per_sample.shape[0]: ", epochs_per_sample.shape[0], ",\t sum_bit_array: ", int(sum_bit_array))
        eps += 1

    # return head_embedding, eps
    return head_embedding, eps, cost / sum_bit_array


@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).

    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample

    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def compute_gradient(similarities, Y, N, D, dY, theta, ee_factor):
    '''
    COMPUTE_GRADIENT 
    ----------
    Compute gradient of Y per each iteration

    Parameters
    ----------
    similarities: (n, n) adjacency matrix (2d array)
    Y: embedded output = (n_components) 2D array
    N: number of rows
    D: input dimension
    dY: gradient
    theta: 
    ee_factor: early exaggeration factor

    Returns
    -------
    None
    '''
    return 0


def evaluate_error(similarities, Y, N, D, theta, ee_factor):
    '''
    EVALUATE ERROR 
    ----------
    evaluate UMAP cost function (approximately)

    Parameters
    ----------
    similarities: (n, n) adjacency matrix (2d array)
    Y: embedded output = (n * n_components) 2D array
    N: number of rows
    D: input dimension
    theta: 
    ee_factor: early exaggeration factor

    Returns
    -------
    C: cost
    '''
    return 0

@numba.njit(fastmath=True)
def progressive_smooth_knn_dist(
    updatedIds,
    k,
    n_iter,
    local_connectivity,
    bandwidth,
    distances,
    table_size,
    rhos,
    sigmas,):
    """Compute a continuous version of the distance to the kth nearest
    neighbor for selected rows. That is, this is similar to knn-distance but
    allows continuous k values rather than requiring an integral k. In essence
    we are simply computing the distance such that the cardinality of fuzzy set
    we generate is k.

    Parameters
    ----------
    updatedIds: list
        Indexes we are goint to update getting sigmas and rhos

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
        
    table_size: integer
        current table size
    
    Returns
    -------
    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    """
    target = np.log2(k) * bandwidth # 2.3192...

    mean_distances = np.mean(distances[:table_size])

    for i in updatedIds:
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ## CALCULATE rhos[i]
        ith_distances = distances[i] # ith_distances.shape = (1, 5)
        non_zero_dists = ith_distances[ith_distances > 0.0] # select elements > 0.0
        if non_zero_dists.shape[0] >= local_connectivity: # non_zero_dists.shape[0] = number of locally connected dots
            index = int(np.floor(local_connectivity)) # local_connectivity = 1.0
            interpolation = local_connectivity - index
            if index > 0:
                rhos[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE: # SMOOTH_K_TOLERANCE = 1e-5
                    rhos[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else: # if index == 0 (0 <= local_connectivity < 1)
                rhos[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rhos[i] = np.max(non_zero_dists)

        ## CALCULATE sigmas[i]
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, k): # range(1,k) => 1,2,...,k-1
                d = distances[i, j] - rhos[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            # break if it is lower than the threshold
            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE: # fabs: Compute the absolute values element-wise.
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rhos[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigmas[i] < MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_distances

    return rhos, sigmas


@numba.njit(fastmath=True)
def progressive_compute_membership_strengths(
    updatedIds,
    indexes,
    distances,
    rhos,
    sigmas,
    n_neighbors,
    rows,
    cols,
    vals,):
    """For selected indexes, construct the membership strength data for the 1-skeleton
    of each local fuzzy simplicial set -- this is formed as a sparse matrix where each
    row is a local fuzzy simplicial set, with a membership strength for the 1-simplex
    to each other data point.

    Parameters
    ----------
    updatedIds: list
        indexes that need to be updated

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """

    for Aid in updatedIds: # point A
        # the neighbors of Aid has been updated
        for Bid in (indexes[Aid]): # point B

            # index of B (e.g., indexes: [0 3 9 2 1] -> ix: [0 1 2 3 4])
            ix = -1
            for i in range(n_neighbors):
                if indexes[Aid][i] == Bid:
                    ix = i
                    break
            if ix == -1:
                raise ValueError("Error retrieving ix value")
            
            if indexes[Aid, ix] == Aid:
                val = 0.0
            elif distances[Aid, ix] - rhos[Aid] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((distances[Aid, ix] - rhos[Aid]) / (sigmas[Aid])))

            rows[Aid * n_neighbors + ix] = Aid
            cols[Aid * n_neighbors + ix] = Bid # indexes[Aid, ix]
            vals[Aid * n_neighbors + ix] = val # sum of the vals = log2(k)*bandwidth

            # print("Aid: {}, Bid: {}, val: {}".format(Aid, Bid, val))

    return rows, cols, vals


# @measure_time
@numba.njit(fastmath=True, parallel=True)
def progressive_optimize_layout(
    head_embedding,
    tail_embedding,
    data,
    head,
    tail,
    epochs,
    last_epochs,
    n_epochs,
    batch_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False):

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]

    epochs_per_sample[epochs_per_sample > batch_epochs ] = batch_epochs
    epoch_of_next_sample = epochs_per_sample.copy()

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()

    run_epoch = int(np.ceil(epoch_of_next_sample.max()))

    for n in range(run_epoch):
        alpha = initial_alpha - float(epochs) * 0.9 / float(last_epochs)
        if alpha < 0.1:
            alpha = 0.1

        if epochs >= n_epochs:
            break

        cost = 0
        bit_array = [0] * epochs_per_sample.shape[0] # numba does not support bitarray, we use list instead

        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n: # loss at 0 epoch because of this

                j = head[i] # first index
                k = tail[i] # second index

                current = head_embedding[j] # position of j index in embedded space
                other = tail_embedding[k] # position of k index in embedded space

                dist_squared = rdist(current, other) # if embedding space is 2D (x1-x2)^2 + (y1-y2)^2

                if dist_squared > 0.0: # if they are not the same pts
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                    if bit_array[i] == 0: # count this edge to the cost
                        c1 = 1.0 / (1.0 + a * pow(dist_squared, 2 * b))
                        cost -= data[i] * math.log(c1)
                        #cost -= math.log(c1)
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d])) # grad_coeff * (y_i - y_j)
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                        if bit_array[i] == 0: # the above edge's negative samples are considered, add this to the cost
                            c2 = a * pow(dist_squared, 2.0 * b) / (1.0 + a * pow(dist_squared, 2.0 * b))
                            cost -= data[i] * gamma * math.log(c2)
                            #cost -= gamma * math.log(c2)
                            if p == (n_neg_samples - 1):
                                bit_array[i] = 1 # In the end, all 1s will be summed up to check the number of edges considered in this epoch
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d])) # grad_coeff * (y_i - y_j)
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        if n != 0:
            sum_bit_array = 1e-4 # prevent 0 division
            for l in range(len(bit_array)):
                sum_bit_array += bit_array[l] # calculate the total number of edges considered in this epoch

        epochs += 1

    return head_embedding, epochs, cost / sum_bit_array


class UMAP(BaseEstimator):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.

    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.

    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.

    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.

    target_metric_kwds: dict (optional, default None)
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.

    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.

    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.

    verbose: bool (optional, default False)
        Controls verbosity of logging.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        n_epochs=None,
        epochs=0,
        last_epochs=200,
        batch_epochs=40,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        metric_kwds=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        ops=1000,
        first_ops=15000,
        verbose=False):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.epochs = epochs
        self.last_epochs = last_epochs
        self.batch_epochs = batch_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.ops = ops
        self.first_ops = first_ops
        self.verbose = verbose

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist must be greater than 0.0")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 2")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 2")
        if not isinstance(self.n_components, int):
            raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer " "larger than 10")
        if self.last_epochs is not None and (
            self.last_epochs <= 10 or not isinstance(self.last_epochs, int)
        ):
            raise ValueError("last_epochs must be a positive integer " "larger than 10")
        if self.batch_epochs < 0:
            raise ValueError("batch_epochs must be positive")
        if self.ops < 0:
            raise ValueError("ops must be positive")
        if (self.first_ops < 0) | (self.first_ops < self.ops):
            raise ValueError("first_ops must be positive and bigger than ops")

    # def fit(self, X, y=None):
    def fit(self, X, y, label, dname):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        self.start = ts()

        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if self.metric_kwds is not None:
            self._metric_kwds = self.metric_kwds
        else:
            self._metric_kwds = {}

        if self.target_metric_kwds is not None:
            self._target_metric_kwds = self.target_metric_kwds
        else:
            self._target_metric_kwds = {}

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))
            
        # Error check n_neighbors based on data size
        if X.shape[0] <= self.n_neighbors:
            if X.shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        if scipy.sparse.isspmatrix_csr(X):
            if not X.has_sorted_indices:
                X.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        # Handle small cases efficiently by computing all distances
        if X.shape[0] < 4096:
            self._small_data = True
            dmat = pairwise_distances(X, metric=self.metric, **self._metric_kwds)
            self.graph_ = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
        else:
            self._small_data = False
            # Standard case
            (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                X,
                self._n_neighbors,
                self.metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.verbose,
            )

            self.graph_ = fuzzy_simplicial_set(
                X,
                self.n_neighbors,
                random_state,
                self.metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )

            self._search_graph = scipy.sparse.lil_matrix(
                (X.shape[0], X.shape[0]), dtype=np.int8
            )
            self._search_graph.rows = self._knn_indices
            self._search_graph.data = (self._knn_dists != 0).astype(np.int8)
            self._search_graph = self._search_graph.maximum(
                self._search_graph.transpose()
            ).tocsr()

            if callable(self.metric):
                self._distance_func = self.metric
            elif self.metric in dist.named_distances:
                self._distance_func = dist.named_distances[self.metric]
            elif self.metric == "precomputed":
                warn(
                    "Using precomputed metric; transform will be unavailable for new data"
                )
            else:
                raise ValueError(
                    "Metric is neither callable, " + "nor a recognised string"
                )

            if self.metric != "precomputed":
                self._dist_args = tuple(self._metric_kwds.values())

                self._random_init, self._tree_init = make_initialisations(
                    self._distance_func, self._dist_args
                )
                self._search = make_initialized_nnd_search(
                    self._distance_func, self._dist_args
                )

        if y is not None:
            if len(X) != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len(X), len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = categorical_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            else:
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    ydmat = pairwise_distances(
                        y_[np.newaxis, :].T,
                        metric=self.target_metric,
                        **self._target_metric_kwds
                    )
                    target_graph = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                else:
                    # Standard case
                    target_graph = fuzzy_simplicial_set(
                        y_[np.newaxis, :].T,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                # product = self.graph_.multiply(target_graph)
                # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
                # #                                        target_graph -
                # #                                        product)
                # self.graph_ = product
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight
                )
                self.graph_ = reset_local_connectivity(self.graph_)
            
        if self.n_epochs is None:
            self.n_epochs = 0

        if self.verbose:
            print(ts(), "Construct embedding")

        self.embedding_ = self.simplicial_set_embedding2(
            self._raw_data,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            init,
            random_state,
            self.metric,
            self._metric_kwds,
            self.verbose,
            self.start,
            label,
            dname,
        )

        if self.verbose:
            print(ts(), " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)
        
        return self


    @measure_time
    def simplicial_set_embedding2(
        self,
        data,
        graph,
        n_components,
        initial_alpha,
        a,
        b,
        gamma,
        negative_sample_rate,
        init,
        random_state,
        metric,
        metric_kwds,
        verbose,
        time,
        label,
        dname):

        graph = graph.tocoo()
        graph.sum_duplicates()
        n_vertices = graph.shape[1]

        if self.n_epochs <= 0:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                self.n_epochs = 50000
            else:
                self.n_epochs = 20000

        graph.data[graph.data < (graph.data.max() / float(self.n_epochs))] = 0.0
        graph.eliminate_zeros()

        if isinstance(init, str) and init == "random":
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come
            initialisation = spectral_layout(
                data,
                graph,
                n_components,
                random_state,
                metric=metric,
                metric_kwds=metric_kwds,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(
                np.float32
            )
        else:
            init_data = np.array(init)
            if len(init_data.shape) == 2:
                if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                    tree = KDTree(init_data)
                    dist, ind = tree.query(init_data, k=2) # (n, k) distance, index
                    nndist = np.mean(dist[:, 1]) # this is just a mean value of the second column. WHY second column??
                    embedding = init_data + random_state.normal(
                        scale=0.001 * nndist, size=init_data.shape
                    ).astype(np.float32) # add a noise 
                else:
                    embedding = init_data

        epochs_per_sample = make_epochs_per_sample(graph.data, self.n_epochs)

        head = graph.row
        tail = graph.col

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64) # 3 random integers

        cost = 0
        self.batch_epochs = 5

        time_taken = ts() - time

        # mkdir if it does not exist
        pathlib.Path(os.path.join(os.getcwd(), 'result', dname, 'umap')).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(os.getcwd(), 'result', dname, 'umap', 'log_fashion.csv'), 'a') as log:
            log.write(f"size,epochs,time_taken,cost\n")
            log.write(f"{embedding.shape[0]},{self.epochs},{time_taken.total_seconds()},{0}\n")

        while self.epochs < self.n_epochs:
            embedding, self.epochs, cost = optimize_layout2(
                embedding,
                embedding,
                graph.data,
                head,
                tail,
                self.epochs,
                self.n_epochs,
                self.batch_epochs,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                rng_state,
                gamma,
                initial_alpha,
                negative_sample_rate,
                verbose=verbose,
            )

            time_taken = ts() - time
            print(f"size: {embedding.shape[0]},\t eps: {self.epochs},\t time taken: {time_taken.total_seconds()},\t cost: {cost}")

            with open(os.path.join(os.getcwd(), 'result', dname, 'umap', 'log_fashion.csv'), 'a') as log:
                log.write(f"{embedding.shape[0]},{self.epochs},{time_taken.total_seconds()},{cost}\n")

            saves = [100, 200, 300, 500, 1000, 2000, 2500]

            if self.epochs in saves:
                with open(os.path.join(os.getcwd(), 'result', dname, 'umap', f'{self.epochs}.csv'), 'wb') as log:
                    np.savetxt(log, embedding, delimiter=",")

                # fig, ax = plt.subplots(1, figsize=(10, 10))
                # plt.scatter(*embedding.T, s=0.3, c=label, cmap='Spectral', alpha=1.0)
                # plt.setp(ax, xticks=[-10, -5, 0, 5, 10], yticks=[-10, -5, 0, 5, 10])
                # plt.ylim(-15.0, +15.0)
                # plt.xlim(-15.0, +15.0)
                # cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
                # cbar.set_ticks(np.arange(10))
                # cbar.set_ticklabels(_item)
                # plt.title('Fashion MNIST Embedded')
                # plt.savefig(os.path.join(os.getcwd(), 'result', dname, 'umap', f'{self.epochs}.png'))

        return embedding

    # PANENE IMPLEMENTATION
    def update_similarity(self, ops, set_op_mix_ratio):
        '''
        UPDATE_SIMILARITY 
        ----------
        1. update KNN Table (i.e., table.run(ops))
        -) compute val_P for points 
            1) newly inserted (ar.addPointResult, # = table.getSize() - ar.addPointResult)
            2) updated (=dirty points, ar.updatePointResult)
        2. collect all ids: 1) & 2) => ar.updatedIds
        3. set initial positions of newly added points (ar.updatePointResult)
            - if epochs == 0, 
            - if epochs > 0, locate them based on the mean value of their neighbors
        4. update perplexities for points in 1) & 2) => ar.updatedIds
        5. make the adjacency matrix symmetric

        Parameters
        ----------
        ops: number of ops
        set_op_mix_ratio: interpolate between (fuzzy) union and intersection

        Returns
        -------
        result: COO format matrix (adjacency matrix)
        '''

        # run for given number of ops
        updates = self.table.run(ops)

        # dirty points (updated IDs of current table compared to the previous table)
        updatedIds = list(updates['updatedIds'])

        for i in range(self.table.size() - updates['addPointResult'], self.table.size()):
            # append newly inserted points
            updatedIds.append(i)

            if self.epochs > 0:
                # overwrite random values with 0
                for j in range(self.n_components):
                    # self.embedding_[i][j] = 0
                    self.embedding_[i][j] = self.embedding_[self.indexes[i][1]][j]

                # set their initial points to the mean of its neighbors
                # for k in range(1, self.n_neighbors):
                #     for j in range(self.n_components):
                #         # issue1: divide by (self.n_neighbors - 1) ?
                #         self.embedding_[i][j] += self.embedding_[self.indexes[i][k]][j] / self.n_neighbors

                # issue2: add random noise after calculation
                if (updates['addPointResult'] > 0) & (i == self.table.size() - 1):
                    nndist = np.mean(self.distances[:, 1]) # WHY second column?? is this just chosen randomly?
                    self.embedding_[self.table.size()-updates['addPointResult']:self.table.size()] += self.random_state.normal(
                        scale=0.001 * nndist, size=[updates['addPointResult'], self.n_components] # scale=0.001? 0.0001?
                    ).astype(np.float32)

        # if self.verbose:
            # for i in updatedIds:
            #     print(f"i: {i}, index: {self.indexes[i]}, distance: {self.distances[i]}, Y[i]: {self.embedding_[i]}")

        '''
        EARLY EXAGGERATION SKIPPED -> use BANDWIDTH in UMAP
        '''
        # compute sigmas and rhos for dirty & newly inserted points
        self.rhos, self.sigmas = progressive_smooth_knn_dist(
            updatedIds,
            self.n_neighbors,
            64, # number of maximum iterations
            self.local_connectivity,
            1.0, # bandwidth
            self.distances,
            self.table.size(),
            self.rhos,
            self.sigmas,)

        # if self.verbose:
        #     print(f"sigmas: {self.sigmas[:self.table.size()]}, rhos: {self.rhos[:self.table.size()]}")

        # compute membership strengths progressively
        self.rows, self.cols, self.vals = progressive_compute_membership_strengths(
            updatedIds,
            self.indexes,
            self.distances,
            self.rhos,
            self.sigmas,
            self.n_neighbors,
            self.rows,
            self.cols,
            self.vals,)

        # COO matrix
        result = scipy.sparse.coo_matrix(
            (self.vals, (self.rows, self.cols)), shape=(self.table.size(), self.table.size())
        )
        result.eliminate_zeros()

        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        # make it symmetric
        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )
        # result = (result + transpose) / 2  # another way to make it symmetric

        result.eliminate_zeros()

        # if self.verbose:
            # print(result)

        return result # return COO matrix


    def run(self, _X, _y, _label, _dname):
        '''
        RUN 
        ----------
        Progressively perform UMAP using PANENE's KNN Table

        1. initialize KNN Table
        2. (randomly) initialize Y
        3. run iteration (max_iter)
            - if size < N (left points are waiting to be added)
                - update similarity matrix using update_similarity()
            - compute gradient & update Y
            - calculate cost (optional)

        Parameters
        ----------
        _X: high-dimensional input
        _y: low-dimensional output = (n * self.n_components) 2D array
        _label: y class (number)
        _dname: dataset name (character)

        Returns
        -------
        None
        '''

        start = ts()

        # check array: the input is checked to be a non-empty 2D array containing only finite values.
        _X = check_array(_X, dtype=np.float32, accept_sparse="csr")
        self._raw_data = _X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if self.metric_kwds is not None:
            self._metric_kwds = self.metric_kwds
        else:
            self._metric_kwds = {}

        if self.target_metric_kwds is not None:
            self._target_metric_kwds = self.target_metric_kwds
        else:
            self._target_metric_kwds = {}

        # if isinstance(self.init, np.ndarray):
        #     init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        # else:
        #     init = self.init

        self.alpha = self._initial_alpha = self.learning_rate

        # Check parameters
        self._validate_parameters()

        if self.verbose:
            print(str(self))
            
        # Error check n_neighbors based on data size
        if _X.shape[0] <= self.n_neighbors:
            if _X.shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self
            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "_X.shape[0] - 1"
            )
            self._n_neighbors = _X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        if scipy.sparse.isspmatrix_csr(_X):
            if not _X.has_sorted_indices:
                _X.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False

        # Set random seed
        self.random_state = check_random_state(self.random_state)

        self.N = _X.shape[0]

        # initialize table & neighbors & distances
        self.indexes = np.zeros((self.N, self.n_neighbors), dtype=np.int64) # with np.in32, it is not optimized
        self.distances = np.zeros((self.N, self.n_neighbors), dtype=np.float32)

        # initialize sigmas and rhos
        self.sigmas = np.zeros(self.N, dtype=np.float32)
        self.rhos = np.zeros(self.N, dtype=np.float32)

        # initialize random Y value following Uniform distribution
        self.embedding_ = np.array(self.random_state.uniform(
            low=-10.0, high=10.0, size=(self.N, self.n_components)
        ).astype(np.float32))

        self.table = KNNTable(_X, self.n_neighbors, self.indexes, self.distances)

        # initialize elements of COO matrix
        self.rows = np.zeros((self.N * self.n_neighbors), dtype=np.int64)
        self.cols = np.zeros((self.N * self.n_neighbors), dtype=np.int64)
        self.vals = np.zeros((self.N * self.n_neighbors), dtype=np.float32)

        # For smaller datasets we can run more epochs
        if self.N <= 10000:
            self.n_epochs = 2000
        else:
            self.n_epochs = 5000

        ###########################
        # self.batch_epochs = 40
        # self.last_epochs = 200
        # self.first_ops = 15000
        # self.ops = 1000
        ###########################

        # mkdir if it does not exist
        pathlib.Path(os.path.join(os.getcwd(), 'result', _dname, 'pumap')).mkdir(parents=True, exist_ok=True)

        # run iteration (work progressively)
        while self.epochs < self.n_epochs:

            if self.table.size() < self.N:
                if self.epochs == 0:
                    self._ops = self.first_ops
                else:
                    self._ops = self.ops
                    self.batch_epochs = 4

                # get COO formatted adjacency matrix
                adj_matrix = self.update_similarity(ops=self._ops, set_op_mix_ratio=1.0)
                self.graph_ = adj_matrix
                if self.epochs == 0:
                    init_time = ts() - start
                    print(f"Finished initialization: {init_time.total_seconds()}")
                    with open(os.path.join(os.getcwd(), 'result', _dname, 'pumap', 'log_fashion.csv'), 'a') as log:
                        log.write(f"size,epochs,time_taken,cost\n")
                        log.write(f"{self.table.size()},{self.epochs},{init_time.total_seconds()},{0}\n")
            else:
                self.batch_epochs = 2

            self.graph_ = self.graph_.tocoo() # type: csr_matrix to coo_matrix
            self.graph_.sum_duplicates()

            # remove values smaller than the threshold (e.g., 1 / 200)
            self.graph_.data[self.graph_.data < (self.graph_.data.max() / float(self.n_epochs))] = 0.0
            self.graph_.eliminate_zeros()

            # head: The indices of the heads of 1-simplices with non-zero membership.
            head = self.graph_.row # first index
            # tail: The indices of the tails of 1-simplices with non-zero membership.
            tail = self.graph_.col # second index

            epochs_per_sample = make_epochs_per_sample(self.graph_.data, self.n_epochs)

            rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64) # 3 random numbers

            # For the first batch, initialize them with spectral embedding
            if self.epochs == 0:
                initialisation = spectral_layout(
                    self._raw_data[:self.table.size()],
                    self.graph_,
                    self.n_components,
                    self.random_state,
                    metric="euclidean",
                    metric_kwds=None,
                )
                expansion = 10.0 / np.abs(initialisation).max()
                embedding = (initialisation * expansion).astype(
                    np.float32
                ) + self.random_state.normal(
                    scale=0.0001, size=[self.graph_.shape[0], self.n_components]
                ).astype(
                    np.float32
                )
                self.embedding_[:self.table.size()] = embedding

            # Optimize embedding progressively
            embedding, self.epochs, cost = progressive_optimize_layout(
                head_embedding=self.embedding_[:self.table.size()],
                tail_embedding=self.embedding_[:self.table.size()],
                data=self.graph_.data,
                head=head,
                tail=tail,
                epochs=self.epochs,
                last_epochs=self.last_epochs,
                n_epochs=self.n_epochs,
                batch_epochs=self.batch_epochs,
                n_vertices = self.graph_.shape[1],
                epochs_per_sample=epochs_per_sample,
                a=self._a,
                b=self._b,
                rng_state=rng_state,
                verbose=True
            )

            self.embedding_[:self.table.size()] = embedding

            time_taken = ts() - start
            print(f"size: {self.table.size()},\t eps: {self.epochs},\t time taken: {time_taken.total_seconds()},\t cost: {cost}")

            with open(os.path.join(os.getcwd(), 'result', _dname, 'pumap', 'log_fashion.csv'), 'a') as log:
                log.write(f"{self.table.size()},{self.epochs},{time_taken.total_seconds()},{cost}\n")

            saves = [100, 200, 300, 500, 1000, 1500, 2000, 2500]

            # per epochs for save
            # save_eps = 100
            # if self.epochs % save_eps == 0:
            if self.epochs in saves:
                with open(os.path.join(os.getcwd(), 'result', _dname, 'pumap', f'{self.epochs}.csv'), 'wb') as log:
                    np.savetxt(log, embedding, delimiter=",")

                # fig, ax = plt.subplots(1, figsize=(10, 10))
                # plt.scatter(*embedding.T, s=0.3, c=_label[:self.table.size()], cmap='Spectral', alpha=1.0)
                # plt.setp(ax, xticks=[-10, -5, 0, 5, 10], yticks=[-10, -5, 0, 5, 10])
                # plt.ylim(-15.0, +15.0)
                # plt.xlim(-15.0, +15.0)
                # # cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
                # # cbar.set_ticks(np.arange(10))
                # # cbar.set_ticklabels(_item)
                # # plt.title('Fashion MNIST Embedded')
                # plt.savefig(os.path.join(os.getcwd(), 'result', _dname, 'pumap', f'{self.epochs}.png'))

        self._input_hash = joblib.hash(self._raw_data)

        return self


    @measure_time
    def fit_transform(self, X, y, label, dname, progressive):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.

        Returns
        -------
        embedding_ : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        if progressive:
            self.run(X, None, label, dname)
            return self.embedding_[:self.table.size()]
        else:
            self.fit(X, None, label, dname)
            return self.embedding_

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        # If we fit just a single instance then error
        if self.embedding_.shape[0] == 1:
            raise ValueError(
                "Transform unavailable when model was fit with"
                "only a single data sample."
            )
        # If we just have the original input then short circuit things
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        x_hash = joblib.hash(X)
        if x_hash == self._input_hash:
            return self.embedding_

        if self._sparse_data:
            raise ValueError("Transform not available for sparse input.")
        elif self.metric == "precomputed":
            raise ValueError(
                "Transform  of new data not available for " "precomputed metric."
            )

        X = check_array(X, dtype=np.float32, order="C")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if self._small_data:
            dmat = pairwise_distances(
                X, self._raw_data, metric=self.metric, **self._metric_kwds
            )
            indices = np.argpartition(dmat, self._n_neighbors)[:, : self._n_neighbors]
            dmat_shortened = submatrix(dmat, indices, self._n_neighbors)
            indices_sorted = np.argsort(dmat_shortened)
            indices = submatrix(indices, indices_sorted, self._n_neighbors)
            dists = submatrix(dmat_shortened, indices_sorted, self._n_neighbors)
        else:
            init = initialise_search(
                self._rp_forest,
                self._raw_data,
                X,
                int(self._n_neighbors * self.transform_queue_size),
                self._random_init,
                self._tree_init,
                rng_state,
            )
            result = self._search(
                self._raw_data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                X,
            )

            indices, dists = deheap_sort(result)
            indices = indices[:, : self._n_neighbors]
            dists = dists[:, : self._n_neighbors]

        adjusted_local_connectivity = max(0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist(
            dists, self._n_neighbors, local_connectivity=adjusted_local_connectivity
        )

        rows, cols, vals = compute_membership_strengths(indices, dists, sigmas, rhos)

        graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # This was a very specially constructed graph with constant degree.
        # That lets us do fancy unpacking by reshaping the csr matrix indices
        # and data. Doing so relies on the constant degree assumption!
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
        weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
        embedding = init_transform(inds, weights, self.embedding_)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = self.n_epochs // 3.0

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col

        embedding = optimize_layout(
            embedding,
            self.embedding_.astype(np.float32, copy=False),  # Fix #179
            head,
            tail,
            n_epochs,
            graph.shape[1],
            epochs_per_sample,
            self._a,
            self._b,
            rng_state,
            self.repulsion_strength,
            self._initial_alpha,
            self.negative_sample_rate,
            verbose=self.verbose,
        )

        return embedding
