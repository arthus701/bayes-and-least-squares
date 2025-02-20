import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patches

from scipy.interpolate import BSpline

from pymaginverse.damping_modules.damping import integrator
from pymaginverse.banded_tools.utils import banded_to_full

from data import (
    signal,
    n_data,
    x_at,
    y_at,
    sigma_o_reported,
    arr,
)
from utils import kernel, paperwidth

SPL_DEGREE = 3
alpha = 0.89
beta = 1.3

fig, axs = plt.subplots(
    1, 3,
    figsize=(paperwidth, 0.3*paperwidth),
    sharex=True,
    sharey=True,
)

for ax in axs:
    ax.set_xlim(np.min(arr), np.max(arr))
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_ylim(-2, 2)
    ax.set_yticks([-1.3, 0, 1.3])

    ax.errorbar(
        x_at,
        y_at,
        yerr=sigma_o_reported,
        ls='',
        color='grey',
        marker='.',
        zorder=2,
        alpha=0.5,
    )

for it in range(2):
    arrow = patches.ConnectionPatch(
        (4.15, 0),
        (-4, 0),
        coordsA=axs[it].transData,
        coordsB=axs[it+1].transData,
        # Default shrink parameter is 0 so can be omitted
        color="black",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

axs[0].plot(
    arr,
    signal(arr),
    zorder=0,
    color='black',
    ls='--',
)

# kernel inversion
cor = kernel(arr, x_at, alpha=alpha, beta=beta)
mat = kernel(x_at, alpha=alpha, beta=beta) + sigma_o_reported**2*np.eye(n_data)
prc = np.linalg.inv(mat)

kernel_solution = cor @ prc @ y_at
axs[1].plot(
    arr,
    kernel_solution,
    zorder=1,
    color='C0',
    lw=2,
)

# spline inversion
# t_step = 1.5
t_step = 2 / 3
knots = np.arange(-9, 9+1, t_step)
design_matrix = BSpline.design_matrix(
    x_at,
    knots,
    SPL_DEGREE,
)
nr_splines = design_matrix.shape[1]
matrix_diag = np.zeros(
    (SPL_DEGREE + 1, nr_splines)
)

for ddt, scale in zip((0, 2), (alpha, beta)):
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
                ddt,
            )
            # place damping in matrix
            matrix_diag[
                it,
                (jt + k):(jt + k + 1)
            ] = scale * spl_integral


dm = design_matrix.toarray()
mat_banded = dm.T @ dm / sigma_o_reported**2 + banded_to_full(matrix_diag)

cfs = np.linalg.inv(mat_banded) @ design_matrix.T @ y_at / sigma_o_reported**2
design_matrix_arr = BSpline.design_matrix(
    arr,
    knots,
    SPL_DEGREE,
)

axs[2].plot(
    arr,
    cfs @ design_matrix_arr.T,
    color='C0',
    lw=2,
    zorder=1,
)
axs[2].plot(
    arr,
    design_matrix_arr.toarray(),
    color='C0',
    lw=1,
    zorder=-1,
    ls='--',
)

fig.tight_layout()
fig.subplots_adjust(wspace=0.2)

fig.savefig(
    '../../fig/approximation_chain.pdf',
    bbox_inches='tight',
    pad_inches=0.,
    transparent=True,
)
