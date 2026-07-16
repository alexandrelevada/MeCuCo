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
from scipy.stats import iqr
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from numpy import log, log2, log10
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
X = skdata.fetch_openml(name='Speech', version=1)
#X = skdata.fetch_openml(name='madelon', version=1)
#X = skdata.fetch_openml(name='har', version=1)
#X = skdata.fetch_openml(name='isolet', version=1)
#X = skdata.fetch_openml(name='parkinson-speech-uci', version=1)
#X = skdata.fetch_openml(name='cnae-9', version=1)
#X = skdata.fetch_openml(name='coil-20', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=1)
#X = skdata.fetch_openml(name='MNIST_784', version=1)
#X = skdata.fetch_openml(name='Fashion-MNIST', version=1)
#X = skdata.fetch_openml(name='Kuzushiji-MNIST', version=1)
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)
#X = skdata.fetch_openml(name='DLBCL', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Lung_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Colon', version=1)
#X = skdata.fetch_openml(name='leukemia', version=1)
#X = skdata.fetch_openml(name='OVA_Breast', version=1)
#X = skdata.fetch_openml(name='GCM', version=1)
#X = skdata.fetch_openml(name='GLI', version=1)
#X = skdata.fetch_openml(name='MLL', version=1)
#X = skdata.fetch_openml(name='SRBCT', version=1)
#X = skdata.fetch_openml(name='hepatitisC', version=1)
#X = skdata.fetch_openml(name='SMK', version=1)


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

# Fast curvature estimation method
start = time.time()
curvatures_f = Mean_Curvatures(dados, nn, n_jobs=-1)
end = time.time()
print('Elapsed time in local curvatures estimation (fast - MeCuCo): ', (end-start))

# # If interquartile range is small compared to the full range, use a log transform to compress curvatures  
# if iqr(curvatures_f)/(curvatures_f.max() - curvatures_f.min()) < 0.1:
#     curvatures_f = log(1 + curvatures_f)

# Curvature normalization
K_f = normalize_curvatures(curvatures_f, 0, 1)  # with normalization
#K_f = curvatures_f                             # without normalization
print('Average curvature \u00B1 Std. Dev.: %.4f \u00B1 %.4f' %(K_f.mean(), K_f.std()))
#print('5% quantile: ', np.quantile(K_f, 0.05))
print('25% quantile: ', np.quantile(K_f, 0.25))
print('50% quantile: ', np.quantile(K_f, 0.5))
print('75% quantile: ', np.quantile(K_f, 0.75))
