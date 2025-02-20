import numpy as np

from matplotlib import pyplot as plt

from scipy.optimize import minimize

from data import (
    n_data,
    x_at,
    y_at,
    sigma_o_reported,
    arr,
)
from utils import kernel, paperwidth

rng = np.random.default_rng(1312)

alpha = 9
beta = 1e-2


def target_misfit(x, level=1.0):
    cor = kernel(x_at, alpha=x[0], beta=x[1])
    mat = kernel(x_at, alpha=x[0], beta=x[1]) \
        + sigma_o_reported**2*np.eye(n_data)
    prc = np.linalg.inv(mat)

    mean = cor @ prc @ y_at
    misfit = np.sqrt(np.mean((y_at-mean)**2)) / sigma_o_reported
    return np.abs(level - misfit)


def neglogmle(x):
    mat = kernel(x_at, alpha=x[0], beta=x[1]) \
        + sigma_o_reported**2*np.eye(n_data)
    prc = np.linalg.inv(mat)

    logdet = np.log(
        np.linalg.det(
            mat
        )
    )
    misfit = - 0.5 * y_at @ prc @ y_at
    return logdet - misfit


res_occam = minimize(
    target_misfit,
    x0=(alpha, beta),
    bounds=np.array(
        [
            [1e-3, None],
            [1e-2, None],
        ]
    )
)
print(res_occam)

res_mle = minimize(
    neglogmle,
    x0=(alpha, beta),
    bounds=np.array(
        [
            [1e-3, None],
            [1e-2, None],
        ]
    )
)
print(res_mle)

fig, axs = plt.subplots(
    2, 3,
    figsize=(paperwidth, 0.3*paperwidth),
    sharex=True,
)
axs[1, 2].sharey(axs[0, 2])
axs[0, 2].set_ylim(0.038, 0.084)
axs[0, 2].set_yticks(np.arange(0.04, 0.09, 0.01))

for it, res in enumerate([res_occam, res_mle]):
    cor = kernel(arr, x_at, alpha=res.x[0], beta=res.x[1])
    mat = kernel(x_at, alpha=res.x[0], beta=res.x[1]) \
        + sigma_o_reported**2*np.eye(n_data)
    prc = np.linalg.inv(mat)

    prior_cov = kernel(arr, alpha=res.x[0], beta=res.x[1])

    pri_chol = np.linalg.cholesky(prior_cov)
    pri_samples = pri_chol @ rng.normal(size=(len(arr), 5))

    mean = cor @ prc @ y_at
    cov = prior_cov - cor @ prc @ cor.T
    std = np.sqrt(np.diag(cov))

    post_chol = np.linalg.cholesky(cov)
    post_samples = mean[:, None] \
        + post_chol @ rng.normal(size=(len(arr), 5))

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
axs[0, 0].set_ylabel("Occam's inversion")
axs[1, 0].set_ylabel('Type-II MLE')
fig.tight_layout()
fig.subplots_adjust(hspace=0.)

fig.savefig(
    '../../fig/parameter_selection_kernel.pdf',
    bbox_inches='tight',
    pad_inches=0.,
    transparent=True,
)
