"""Utility functions."""

from .base import mldivide, mrdivide

from .covariances import (
    block_covariance,
    convmtx,
    cov_lags,
    nonlinear_eigenspace,
    pca,
    regcov,
    tscov,
    tsxcov,
)
from .denoise import (
    demean,
    find_outlier_samples,
    find_outlier_trials,
    mean_over_trials,
    wpwr,
)
from .matrix import (
    fold,
    matmul3d,
    multishift,
    multismooth,
    normcol,
    relshift,
    shift,
    shiftnd,
    sliding_window,
    theshapeof,
    unfold,
    unsqueeze,
    widen_mask,
)
from .sig import (
    gaussfilt,
    hilbert_envelope,
    slope_sum,
    smooth,
    spectral_envelope,
    teager_kaiser,
)

