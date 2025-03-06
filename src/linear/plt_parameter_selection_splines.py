import numpy as np
import arviz as az

from matplotlib import pyplot as plt

from scipy.interpolate import BSpline

from pymaginverse.damping_modules.damping import integrator
from pymaginverse.banded_tools.utils import banded_to_full

from data import (
    x_at,
    y_at,
    sigma_o_reported,
    arr,
)
from utils import paperwidth

SPL_DEGREE = 3
rng = np.random.default_rng(1312)

t_step = 2 / 3
knots = np.arange(-9, 9+1, t_step)
design_matrix = BSpline.design_matrix(
    x_at,
    knots,
    SPL_DEGREE,
)
nr_splines = design_matrix.shape[1]

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


dm = design_matrix.toarray()
design_matrix_at = BSpline.design_matrix(
    x_at,
    knots,
    SPL_DEGREE,
)


fig, axs = plt.subplots(
    2, 3,
    figsize=(paperwidth, 0.3*paperwidth),
    sharex=True,
)
axs[1, 2].sharey(axs[0, 2])
axs[0, 2].set_ylim(0.038, 0.084)
axs[0, 2].set_yticks(np.arange(0.04, 0.09, 0.01))

design_matrix_arr = BSpline.design_matrix(
    arr,
    knots,
    SPL_DEGREE,
)
for it in range(2):
    if it == 0:
        a = 1.778
        b = 2.371e-1
        C = a * S + b * T
        mat_banded = dm.T @ dm / sigma_o_reported**2 + C
        pst_cov = np.linalg.inv(mat_banded)
        cfs = pst_cov @ design_matrix.T @ y_at \
            / sigma_o_reported**2

        mean = cfs @ design_matrix_arr.T
        std = np.sqrt(
            np.einsum(
                'ij, jk, ik -> i',
                design_matrix_arr.toarray(),
                pst_cov,
                design_matrix_arr.toarray(),
            )
        )
        post_chol = np.linalg.cholesky(pst_cov)
        samples = cfs[:, None] + post_chol @ rng.normal(size=(nr_splines, 5))
        post_samples = design_matrix_arr @ samples

        pri_chol = np.linalg.cholesky(np.linalg.inv(C))
        samples = pri_chol @ rng.normal(size=(nr_splines, 5))
        pri_samples = design_matrix_arr @ samples
    else:
        prior_idata = az.from_netcdf(
            '../../out/prior_samples_splines.nc',
        )
        cfs = (
            prior_idata
            .posterior['spline_coeffs']
            .values
            .reshape(-1, nr_splines).T
        )

        pri_samples = design_matrix_arr @ cfs
        pri_samples = pri_samples[:, [21, 343, 112, 101, 4]]

        posterior_idata = az.from_netcdf(
            '../../out/posterior_samples_splines.nc',
        )
        summary = az.summary(posterior_idata)
        print(summary.loc[['a', 'b'], :])
        cfs = (
            posterior_idata
            .posterior['spline_coeffs']
            .values
            .reshape(-1, nr_splines)
        )
        post_samples = design_matrix_arr @ cfs.T
        mean = np.mean(post_samples, axis=1)
        std = np.std(post_samples, axis=1)
        post_samples = post_samples[:, :5]

    axs[it, 0].plot(
        arr,
        design_matrix_arr.toarray(),
        color='C0',
        lw=1,
        zorder=-1,
        ls='--',
    )
    axs[it, 1].plot(
        arr,
        design_matrix_arr.toarray(),
        color='C0',
        lw=1,
        zorder=-1,
        ls='--',
    )
    axs[it, 0].set_xlim(np.min(arr), np.max(arr))
    axs[it, 0].set_ylim(-1.8, 1.8)
    axs[it, 0].set_yticks([-1.3, 0., 1.3])
    axs[it, 1].set_ylim(-1.8, 1.8)
    axs[it, 1].set_yticks([-1.3, 0., 1.3])

    axs[it, 0].plot(
        arr,
        pri_samples[:, 1:],
        color='C0',
        lw=1,
        alpha=0.5,
        zorder=-1,
    )
    axs[it, 0].plot(
        arr,
        pri_samples[:, 0],
        color='C0',
        lw=2,
        zorder=1,
    )

    axs[it, 1].errorbar(
        x_at,
        y_at,
        yerr=sigma_o_reported,
        ls='',
        color='grey',
        marker='.',
        zorder=2,
        alpha=0.5,
    )
    axs[it, 1].plot(
        arr,
        mean,
        color='C0',
        lw=2,
        zorder=4,
    )
    axs[it, 1].plot(
        arr,
        post_samples,
        color='C0',
        lw=1,
        alpha=0.5,
        zorder=-1,
    )

    axs[it, 2].plot(
        arr,
        std,
        color='C0',
        lw=2,
        zorder=1,
    )

    # std_mean = std[10:-10].mean()
    std_mean = std.mean()
    axs[it, 2].axhline(
        std_mean,
        color='C0',
        lw=1,
        zorder=0,
        ls='--'
    )
    # axs[it, 2].annotate(
    #     f'{std_mean:.3f}',
    #     (-3, std_mean-0.01),
    #     color='C0',
    # )

axs[0, 0].set_title('Prior samples')
axs[0, 1].set_title('Posterior samples')
axs[0, 2].set_title('Posterior standard deviation')
axs[0, 0].set_ylabel('L-curve')
axs[1, 0].set_ylabel('Hierarchical')
fig.tight_layout()
fig.subplots_adjust(hspace=0.)

fig.savefig(
    '../../fig/parameter_selection_splines.pdf',
    bbox_inches='tight',
    pad_inches=0.,
    transparent=True,
)
