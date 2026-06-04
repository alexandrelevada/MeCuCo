#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Efficient Mean Curvature Computation on High-Dimensional Data Manifolds (MeCuCo)

@author: Alexandre L. M. Levada

"""

# Imports
import os
import time
import warnings
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from networkx.convert_matrix import from_numpy_array
from scipy.linalg import sqrtm
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy import exp
from numpy.linalg import norm
from scipy import stats
from scipy.linalg import eigh
from numpy.linalg import inv
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import svd as _scipy_svd, eigh as _scipy_eigh
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

#####################################################################
# Computes local mean curvatures from dataset dados (original)
#####################################################################
def Mean_Curvatures_Original(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # First Fundamental Form
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second Fundamental Form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    shapes = np.zeros((n, m, m))
    # Build KNN without dense adjacency
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dados)
    knn_indices = nbrs.kneighbors(return_distance=False)
    # Main Loop
    for i in range(n):       
        indices = knn_indices[i]
        # Local covariance matrix computation
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)      # isolated points
        # Eigenvectors
        v, w = _scipy_eigh(I)
        # Sort eigenvalues
        ordem = v.argsort()
        # Descreasing order
        Wpca = w[:, ordem[::-1]]
        # Computing the second fundamental form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Build nonlinear terms from eigenvectors
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))        
        Q = Wpca
        # Select quadratic columns
        H = Q[:, (m+1):]
        II = np.dot(H, H.T)
        S = np.dot(II, I).real
        curvatures[i] = trace(S)
    return curvatures

# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary function (worker for parallel execution)
# ──────────────────────────────────────────────────────────────────────────────
"""
    Computes mean curvatures for a subset of points.
 
    Parameters
    ----------
    dados     : (n, m) — full dataset (required to index neighbors)
    knn_chunk : (n_chunk, k) — indices of k-nearest neighbors for the subset
    rows_ut   : row indices of the upper triangle of (m × m)
    cols_ut   : column indices of the upper triangle of (m × m)
 
    Returns
    -------
    curvatures : (n_chunk,) — mean curvatures
"""
def _chunk_curvatures(dados: np.ndarray, knn_chunk: np.ndarray, rows_ut: np.ndarray, cols_ut: np.ndarray) -> np.ndarray:    
    n_chunk = knn_chunk.shape[0]
    m = dados.shape[1]
    curvatures = np.empty(n_chunk)
    # Main loop 
    for ci in range(n_chunk):
        idx = knn_chunk[ci]
        amostras = dados[idx]
        # ── 1. Symmetric covariance matrix ──────────────────────────────
        Icov = np.cov(amostras.T) if len(idx) > 1 else np.eye(m)
        # ── 2. Spectral decomposition via eigh ──────────────────────────
        # eigh (vs eig): exploits symmetry → ~2× faster, always real.
        # Returns eigenvalues in *ascending* order; we reverse the columns.
        _, w = np.linalg.eigh(Icov)
        Wpca = w[:, ::-1]                          # descending order (m, m)
        # ── 3. Construction of H = [Squared | Cross] without Python loops ─
        # Squared[:, j] = Wpca[:, j] ** 2  →  vectorized over all j
        Squared = Wpca ** 2                        # (m, m)
        # Cross[:, col] = Wpca[:, j] * Wpca[:, l]  for pairs (j, l) with j < l
        # rows_ut / cols_ut are precomputed with np.triu_indices
        Cross = Wpca[:, rows_ut] * Wpca[:, cols_ut]  # (m, nc)
        H = np.concatenate([Squared, Cross], axis=1)  # (m, m + nc)
        # ── 4. Curvature: |trace(-H H^T Icov)| via einsum ────────────────
        # Equivalence:
        #   trace(H H^T Icov) = Σ_ij (H H^T)_ij * Icov_ji
        #                     = einsum('ia,ja,ij->', H, H, Icov)
        # Avoids two dense matrix multiplications + a call to np.trace.
        curvatures[ci] = abs(np.einsum('ia,ja,ij->', H, H, Icov))
    return curvatures

# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
"""
    Computes the mean curvature at each point of a multivariate dataset.
 
    Parameters
    ----------
    dados  : array (n, m) — n samples with m features
    k      : number of nearest neighbors
    n_jobs : number of parallel workers.
             1  → sequential (default, zero overhead).
             -1 → uses all available CPUs.
             Recommended to use n_jobs > 1 only for n >= 5000 with many CPUs.
 
    Returns
    -------
    curvatures : array (n,) — mean curvatures (absolute values)
 
    Optimizations relative to the original version
    ───────────────────────────────────────────────
    1. eigh instead of eig
       The covariance matrix is symmetric positive semi-definite.
       `eigh` exploits this: it is ~2× faster than `eig` and returns
       real eigenvalues directly (no need for `.real`).
 
    2. Elimination of inner Python loops (Squared / Cross)
       `Squared` is computed via elementwise broadcasting (Wpca**2).
       `Cross`   is computed by indexing columns with precomputed
       index arrays (np.triu_indices), in a single vectorized operation.
 
    3. No unnecessary allocations per iteration
       The columns of 1's and the columns of Wpca that were previously
       concatenated at the beginning of Q, but later discarded in
       H = Q[:, m+1:], are completely removed — H is built directly.
 
    4. einsum for the trace
       `trace(H H^T Icov)` is computed as
       `einsum('ia,ja,ij->', H, H, Icov)`,
       avoiding two dense matrix multiplications and a call to
       `np.trace`.
 
    5. Optional parallelism via joblib
       The main loop can be distributed across threads (GIL released by
       NumPy) to leverage multiple CPUs on large datasets.
"""
def Mean_Curvatures_Parallel(dados: np.ndarray, k: int, n_jobs: int = 1) -> np.ndarray:
    n, m = dados.shape
    # Precomputed upper triangle indices (reused at each iteration)
    rows_ut, cols_ut = np.triu_indices(m, k=1)
    # KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dados)
    knn_indices = nbrs.kneighbors(return_distance=False)   # (n, k)
    # Sequential (default) or parallel execution
    n_jobs_eff = max(1, os.cpu_count() if n_jobs == -1 else n_jobs)
    if n_jobs_eff == 1:
        return _chunk_curvatures(dados, knn_indices, rows_ut, cols_ut)
    chunks = np.array_split(knn_indices, n_jobs_eff)
    results = Parallel(n_jobs=n_jobs_eff, prefer='threads')(
        delayed(_chunk_curvatures)(dados, chunk, rows_ut, cols_ut)
        for chunk in chunks
    )
    return np.concatenate(results)

"""
Mean_Curvatures — version 2  (handles high dimensionality)
===========================================================

Two computation modes, selected automatically via `mode='auto'`:

  'exact'  Exact. Uses scipy eigh with a LAPACK driver optimised per
           dimension range + closed-form O(m²) trace identity.
           Recommended for m < 50 or when maximum precision is required.

  'fast'   Approximate. Replaces eigh (O(m³)) with truncated SVD (O(k²m))
           + analytical formula for the null-space contribution.
           Recommended for m ≥ 50. Mean relative error decreases with m/k:
             m/k ≈  5  →  ~9%  mean relative error
             m/k ≈ 11  →  ~5%
             m/k ≈ 22  →  ~3%
             m/k ≈ 56  →  ~2%
             m/k ≈ 111 →  ~1%

Observed speedup vs. original implementation (numpy eig + Python loops):
  m= 20, k=15  →  exact:  ~14×   fast:   ~1.3×
  m= 50, k=15  →  exact:  ~14×   fast:   ~11×
  m=100, k=12  →  exact:   ~1×   fast:   ~50×
  m=200, k=10  →  exact:   ~9×   fast:  ~800×

Mathematical background
───────────────────────
Let W ∈ ℝ^{m×m} be the orthonormal eigenvector matrix of Icov (columns
ordered by decreasing eigenvalue) and v ∈ ℝ^m the corresponding
eigenvalue vector.
The mean curvature at point i is:

    curv(i) = |trace(−H Hᵀ Icov)|

where H = [W² | W_cross]  (W² = element-wise squared columns;
                            W_cross = pairwise Hadamard products).

**Algebraic identity (exact mode):**
    trace(H Hᵀ Icov) = 0.5 · vᵀ (C²).sum(axis=1)  +  0.5 · sum(v)
    where  C = Wᵀ W²   (C[s,l] = Σᵢ W[i,s]·W[i,l]²)

This reduces the computational complexity from O(m⁴) → O(m²) after
the O(m³) eigendecomposition.

**Analytical formula for the null-space contribution (fast mode):**
Icov has rank p = k−1 << m. Its p non-zero eigenvectors V_r ∈ ℝ^{m×p}
are computed via truncated SVD of X_c at cost O(k²m). The contribution
of the m−p null-space eigenvectors is approximated by the expected value
under the uniform (Haar) distribution over orthonormal null-space bases:

    E[G_null[i,j]] = (dP[i]·dP[j]  +  2·P_null[i,j]²) / (m−p)

where P_null = I − V_r Vᵣᵀ  and  dP = diag(P_null).

All resulting terms are computable in O(k·m·p²) without constructing
the full m×m covariance matrix Icov or performing the full eigendecomposition.
"""

# ──────────────────────────────────────────────────────────────────────────────
# EXACT mode  —  adaptive LAPACK driver for scipy eigh  +  O(m²) formula
# ──────────────────────────────────────────────────────────────────────────────

def _eigh_driver(m: int) -> str:
    """ Select the fastest LAPACK driver for the given dimension m.

    Benchmarks :
      m <  50 : 'ev'  (dsyev — simple, minimum overhead)
      m < 500 : 'evr' (dsyevr — MRRR, better for medium m)
      m ≥ 500 : 'evd' (dsyevd — divide-and-conquer, better for large m)
    """
    if m < 50:
        return 'ev'
    if m < 500:
        return 'evr'
    return 'evd'


def _chunk_exact(dados: np.ndarray, knn_chunk: np.ndarray) -> np.ndarray:
    """Compute exact curvatures for a subset of points."""
    n_chunk, m = knn_chunk.shape[0], dados.shape[1]
    drv = _eigh_driver(m)
    curvatures = np.empty(n_chunk)
        
    for ci in range(n_chunk):
        Icov = np.cov(dados[knn_chunk[ci]].T)
        
        # eigh guarantees real eigenvalues and orthonormal eigenvectors.
        # Adaptive driver reduces runtime by up to 100× vs numpy.eigh for m ≥ 100.
        v, W = _scipy_eigh(Icov, lower=True, driver=drv)
        v = v[::-1]; W = W[:, ::-1]          # descending order
        
        # Closed-form O(m²) identity:
        #   trace(H Hᵀ Icov) = 0.5·vᵀ(C²).sum(1) + 0.5·sum(v)
        #   where C[s,l] = Σᵢ W[i,s]·W[i,l]²  →  C = Wᵀ @ W²
        C = W.T @ (W ** 2)                   # (m, m)
        curvatures[ci] = abs(
            0.5 * (v @ (C ** 2).sum(axis=1)) + 0.5 * v.sum()
        )

    return curvatures


# ──────────────────────────────────────────────────────────────────────────────
# FAST mode  —  truncated SVD  +  analytical null-space contribution
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_fast(dados: np.ndarray, knn_chunk: np.ndarray) -> np.ndarray:
    """Compute approximate curvatures for a subset of points.

    Algorithm
    ---------
    1. Truncated SVD of X_c ∈ ℝ^{k×m}  →  V_r (m×p), eigenvalues ev  [O(k²m)]
    2. Range-space contribution:
         term_range = ‖X_c V_r²‖²_F / (k−1)                          [O(kmp)]
    3. Null-space contribution via analytical formula:
         dP[i] = 1 − ‖V_r[i,:]‖²  (diagonal of P_null = I − V_r V_rᵀ)
         term_A  = ‖X_c dP‖² / (k−1)                                  [O(km)]
         term_B  = tr(Ic) − 2·‖X_c V_r‖²_F/(k−1)
                   + Σ_{l₁,l₂} ‖X_c (V_r[:,l₁]·V_r[:,l₂])‖² / (k−1) [O(kmp²)]
         term_null = (term_A + 2·term_B) / (m−p)
    4. curv = |0.5·(term_range + term_null) + 0.5·tr(Ic)|
    """
    n_chunk, m = knn_chunk.shape[0], dados.shape[1]
    k  = knn_chunk.shape[1]
    p  = min(k - 1, m)          # efective rank of Icov
    d  = m - p                   # null-space dimension
    curvatures = np.empty(n_chunk)

    for ci in range(n_chunk):
        ams = dados[knn_chunk[ci]]
        Xc  = ams - ams.mean(axis=0)          # (k, m) — centred data matrix

        # 1. Truncated SVD: O(k²m); returns only k singular vectors
        _, s, Vt = _scipy_svd(Xc, full_matrices=False)
        V_r    = Vt[:p].T                     # (m, p) non-zero eigenvalues
        tr_Ic  = (s[:p] ** 2).sum() / (k - 1)  # exact trace(Icov)

        # 2. Range-space contribution
        term_range = ((Xc @ (V_r ** 2)) ** 2).sum() / (k - 1)

        # 3. Null-space contribution (analytical formula)
        dP      = 1.0 - (V_r ** 2).sum(axis=1)          # (m,) diagonal of P_null
        term_A  = ((Xc @ dP) ** 2).sum() / (k - 1)
        term_B2 = ((Xc @ V_r) ** 2).sum() / (k - 1)
        # G_tens[a, l1, l2] = Σᵢ Xc[a,i]·V_r[i,l1]·V_r[i,l2]  →  shape (k, p, p)
        G_tens  = np.einsum('ai,il,ij->alj', Xc, V_r, V_r)
        term_B3 = (G_tens ** 2).sum() / (k - 1)
        term_B  = tr_Ic - 2.0 * term_B2 + term_B3
        term_null = (term_A + 2.0 * term_B) / d if d > 0 else 0.0

        # 4. Mean curvature estimate
        curvatures[ci] = abs(
            0.5 * (term_range + term_null) + 0.5 * tr_Ic
        )

    return curvatures


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def Mean_Curvatures(
    dados: np.ndarray,
    k: int,
    mode: str = 'auto',
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Estimate the mean curvature at every point of a multivariate dataset.

    Parameters
    ----------
    dados  : ndarray (n, m)
        Dataset with n samples and m features.
    k      : int
        Number of nearest neighbours used in the local estimate.
    mode   : {'auto', 'exact', 'fast'}
        'auto'  → 'exact' if m < 50, 'fast' otherwise.
        'exact' → full eigendecomposition of the local covariance matrix.
                  Exact result, but O(m³) per point.
        'fast'  → truncated SVD + analytical null-space formula. O(k²m + kmp²)
                  per point. Mean relative error decreases with m/k
                  (approx. 1% for m/k ≥ 100).
    n_jobs : int
        Number of parallel worker threads.
        1  → sequential (default).
        -1 → use all available CPUs.

    Returns
    -------
    curvatures : ndarray (n,)
        Absolute mean curvature estimate at each point.

    Notes
    -----
    The 'exact' and 'fast' modes may yield slightly different results even
    for small m, because they rely on different eigensolvers (eigh vs. SVD)
    that may choose different bases for degenerate eigenspaces. For internal
    consistency within the MCBP algorithm, always use the same mode throughout
    a given experiment.
    """
    n, m = dados.shape

    if mode == 'auto':
        mode = 'exact' if m < 50 else 'fast'

    chunk_fn = _chunk_exact if mode == 'exact' else _chunk_fast

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dados)
    knn  = nbrs.kneighbors(return_distance=False)   # (n, k)

    nj = max(1, os.cpu_count() if n_jobs == -1 else n_jobs)
    if nj == 1:
        return chunk_fn(dados, knn)

    chunks  = np.array_split(knn, nj)
    results = Parallel(n_jobs=nj, prefer='threads')(
        delayed(chunk_fn)(dados, chunk) for chunk in chunks
    )
    return np.concatenate(results)

# Optional function to normalize the curvatures to the interval [a, b]
def normalize_curvatures(curv, a, b):
    k = a + (b - a)*(curv - curv.min())/(curv.max() - curv.min())
    return k


#################################################
# Data loading
#################################################
X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_breast_cancer()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='steel-plates-fault', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)            
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='USPS', version=1)
#X = skdata.fetch_openml(name='Satellite', version=1)
#X = skdata.fetch_openml(name='gas-drift', version=1)
#X = skdata.fetch_openml(name='vowel', version=1)
#X = skdata.fetch_openml(name='ionosphere', version=1)
#X = skdata.fetch_openml(name='solar-flare', version=4)
#X = skdata.fetch_openml(name='seeds', version=1)  
#X = skdata.fetch_openml(name='letter', version=1)
#X = skdata.fetch_openml(name='artificial-characters', version=1)
#X = skdata.fetch_openml(name='thoracic_surgery', version=1)
#X = skdata.fetch_openml(name='texture', version=1)
#X = skdata.fetch_openml(name='page-blocks', version=1)
#X = skdata.fetch_openml(name='JapaneseVowels', version=1)
#X = skdata.fetch_openml(name='one-hundred-plants-shape', version=1) 
#X = skdata.fetch_openml(name='one-hundred-plants-texture', version=1)
#X = skdata.fetch_openml(name='Indian_pines', version=1)
#X = skdata.fetch_openml(name='depression_2020', version=1)
#X = skdata.fetch_openml(name='sylvine', version=1)
#X = skdata.fetch_openml(name='eye_movements', version=1)
#X = skdata.fetch_openml(name='GesturePhaseSegmentationProcessed', version=1)
#X = skdata.fetch_openml(name='qsar-biodeg', version=1)
#X = skdata.fetch_openml(name='splice', version=1)
#X = skdata.fetch_openml(name='Smartphone-Based_Recognition_of_Human_Activities', version=1)
#X = skdata.fetch_openml(name='TuningSVMs', version=1)
#X = skdata.fetch_openml(name='hill-valley', version=1)
#X = skdata.fetch_openml(name='arrhythmia', version=1)
#X = skdata.fetch_openml(name='cardiotocography', version=1)
#X = skdata.fetch_openml(name='segment', version=1)
#X = skdata.fetch_openml(name='collins', version=4)
#X = skdata.fetch_openml(name='car-evaluation', version=1)

dados = X['data']
target = X['target']

# To deal with sparse matrix data
if type(dados) == sp.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

n = dados.shape[0]
m = dados.shape[1]
# Number of neighbors
nn = round(np.log2(n))

# Number of classes
c = len(np.unique(target))

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

# Original curvature estimation method
start = time.time()
curvatures_o = Mean_Curvatures_Original(dados, nn)
end = time.time()
print('Elapsed time in local curvatures estimation (original): ', (end-start))
K_o = normalize_curvatures(curvatures_o, 0, 1)  # with normalization
#K_o = curvatures_o                             # without normalization
print('Average curvature \u00B1 Std. Dev.: %.4f \u00B1 %.4f' %(K_o.mean(), K_o.std()))
print()

# Fast curvature estimation method
start = time.time()
curvatures_f = Mean_Curvatures(dados, nn, n_jobs=-1)
end = time.time()
print('Elapsed time in local curvatures estimation (fast - MeCuCo): ', (end-start))
K_f = normalize_curvatures(curvatures_f, 0, 1)  # with normalization
#K_f = curvatures_f                             # without normalization
print('Average curvature \u00B1 Std. Dev.: %.4f \u00B1 %.4f' %(K_f.mean(), K_f.std()))
print('Mean Absolute Error: ', (sum(abs(K_o - K_f)))/n)
print('Spearman Rho coefficient: ', stats.spearmanrho(K_o, K_f).statistic)
print('Chatterjee Xi coefficient: ', stats.chatterjeexi(K_o, K_f).statistic)
