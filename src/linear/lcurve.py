import numpy as np

from tqdm import tqdm

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

SPL_DEGREE = 3
alpha = 0.89
beta = 1.3

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
print(nr_splines)
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

initial_beta = 1

a_s = np.logspace(-3, 2, num=41)
b_s = np.logspace(-3, 2, num=41)
print(a_s)

A_s, B_s = np.meshgrid(a_s, b_s)
ab_s = np.array(
    [
        A_s.flatten(),
        B_s.flatten(),
    ]
).T

norms = []
misfits = []

fig, ax = plt.subplots(
    1, 1,
    figsize=(5, 5)
)
for a, b in tqdm(ab_s):
    C = a * S + b * T
    mat_banded = dm.T @ dm / sigma_o_reported**2 + C
    cfs = (
        np.linalg.inv(mat_banded)
        @ design_matrix.T
        @ y_at / sigma_o_reported**2
    )
    norm = np.sqrt(cfs @ np.linalg.inv(C) @ cfs)
    prediction = cfs @ design_matrix_at.T
    misfit = np.sqrt(np.mean((y_at-prediction)**2)) / sigma_o_reported
    norms.append(norm)
    misfits.append(misfit)
    # ax.annotate(
    #     f'({a:.3e}, {b:.3e})',
    #     (misfit, norm),
    # )


ax.scatter(
    misfits,
    norms,
)
ax.set_xscale('log')
ax.set_yscale('log')
# xlim = ax.get_xlim()
# ax.set_xlim(1.8, 2.2)
# ylim = ax.get_ylim()
# ax.set_ylim(0.18, 2.2)

fix_alpha = 1.778
fix_beta = 2.371e-1

C = fix_alpha * S + fix_beta * T

fig_3, ax_3 = plt.subplots(
    1, 1,
    figsize=(5, 5)
)

mat_banded = dm.T @ dm / sigma_o_reported**2 + C

cfs = np.linalg.inv(mat_banded) @ design_matrix.T @ y_at / sigma_o_reported**2
design_matrix_arr = BSpline.design_matrix(
    arr,
    knots,
    SPL_DEGREE,
)

ax_3.set_xlim(np.min(arr), np.max(arr))
ax_3.set_xticks([])
ax_3.set_ylim(-2, 2)
ax_3.set_yticks([])
ax_3.errorbar(
    x_at,
    y_at,
    yerr=sigma_o_reported,
    ls='',
    color='grey',
    marker='.',
    zorder=2,
    alpha=0.5,
)
ax_3.plot(
    arr,
    cfs @ design_matrix_arr.T,
    color='C0',
    lw=2,
    zorder=1,
)
ax_3.plot(
    arr,
    design_matrix_arr.toarray(),
    color='C0',
    lw=1,
    zorder=-1,
    ls='--',
)

fig_4, ax_4 = plt.subplots(
    1, 1,
    figsize=(5, 5)
)

rng = np.random.default_rng(1312)
chol = np.linalg.cholesky(np.linalg.inv(C))
samples = chol @ rng.normal(size=(nr_splines, 5))
curves = design_matrix_arr @ samples

ax_4.plot(
    arr,
    curves[:, 1:],
    color='C0',
    lw=1,
    alpha=0.5,
    zorder=-1,
)
ax_4.plot(
    arr,
    curves[:, 0],
    color='C0',
    lw=2,
    zorder=1,
)


plt.show()
