import numpy as np

from scipy.interpolate import BSpline

import pymc as pm
from pytensor import tensor as pt

from pymc.sampling import jax as pmj

from pymaginverse.damping_modules.damping import integrator
from pymaginverse.banded_tools.utils import banded_to_full

from data import (
    n_data,
    x_at,
    y_at,
    sigma_o_reported,
)

SPL_DEGREE = 3

t_step = 2 / 3
knots = np.arange(-9, 9+1, t_step)
design_matrix_at = BSpline.design_matrix(
    x_at,
    knots,
    SPL_DEGREE,
).toarray()
nr_splines = design_matrix_at.shape[1]

S = np.zeros(
    (SPL_DEGREE + 1, nr_splines)
)
for it in range(SPL_DEGREE + 1):
    # k takes care of the correct position in the banded format.
    k = SPL_DEGREE - it
    for jt in range(nr_splines - k):
        # integrate cubic B-Splines
        spl_integral = integrator(
            jt,
            jt + k,
            nr_splines,
            t_step,
            0,
        )
        # place damping in matrix
        S[
            it,
            (jt + k):(jt + k + 1)
        ] = spl_integral
S = banded_to_full(S)


T = np.zeros(
    (SPL_DEGREE + 1, nr_splines)
)
for it in range(SPL_DEGREE + 1):
    # k takes care of the correct position in the banded format.
    k = SPL_DEGREE - it
    for jt in range(nr_splines - k):
        # integrate cubic B-Splines
        spl_integral = integrator(
            jt,
            jt + k,
            nr_splines,
            t_step,
            2,
        )
        # place damping in matrix
        T[
            it,
            (jt + k):(jt + k + 1)
        ] = spl_integral
T = banded_to_full(T)
with pm.Model() as mcModel:
    log_a = pm.Normal(
        'log_a',
        mu=1.,
        sigma=2.,
    )
    a = pm.Deterministic(
        'a',
        pm.math.exp(log_a),
    )
    log_b = pm.Normal(
        'log_b',
        mu=0.,
        sigma=5.,
    )
    b = pm.Deterministic(
        'b',
        pm.math.exp(log_b),
    )
    C = a * S + b * T
    prior_chol = pt.slinalg.cholesky(C)
    cfs_centered = pm.Normal(
        'spline_coeffs_centered',
        mu=0.,
        sigma=1.,
        size=nr_splines,
    )
    cfs = pm.Deterministic(
        'spline_coeffs',
        prior_chol @ cfs_centered,
    )

    prediction = cfs @ design_matrix_at.T
    res = pm.Deterministic(
        'res',
        (y_at - prediction) / sigma_o_reported
    )
    obs = pm.Normal(
        'obs',
        mu=res,
        sigma=1.,
        observed=np.zeros(n_data),
    )
with mcModel:
    idata = pmj.sample_numpyro_nuts(
        500,
        tune=1000,
        progressbar=True,
        chains=4,
        target_accept=0.95,
        random_seed=161,
        postprocessing_backend='cpu',
    )
    idata.to_netcdf(
        '../../out/posterior_samples_splines.nc'
    )
