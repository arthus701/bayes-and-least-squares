import numpy as np

from pandas import read_table

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scipy.interpolate import BSpline

from pymagglobal.utils import scaling
from pymagglobal.core import _power_spectrum as power_spectrum

from utils import get_ps_from_samples

paperwidth = 46.9549534722518 / 4

fig, ax = plt.subplots(
    1, 1,
    figsize=(0.4 * paperwidth, 0.4 * paperwidth))

ax.set_xlabel(r'degree $\ell$')
ax.set_ylabel(r'r$_\sigma$ [$\mu$T$^2$]')

igrf = read_table(
    'https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf13coeffs.txt',
    comment='#',
    header=[0, 1],
    sep=r'\s+',
    index_col=[0, 1, 2],
)

ls_igrf = igrf.index.get_level_values(1).values

pws_list = []
scl = scaling(6371.2, 3485, 13)
for _, series in igrf[['DGRF', 'IGRF']].items():
    pws_list.append(power_spectrum(ls_igrf, series.values / 1e3 * scl)[0:10])

ls = np.arange(10) + 1
ref_powers = np.vstack(pws_list).T

# ax.plot(
#     ls,
#     ref_powers,
#     color='grey',
#     zorder=0,
#     alpha=0.5,
# )

with np.load('../out/arch10k_samples.npz') as fh:
    t_base = BSpline.design_matrix(
        fh['knots'][3:-3],
        fh['knots'],
        3
    ).toarray()
    prior_10k = np.einsum(
        'ij, j...->i...',
        t_base,
        fh['prior'],
    )
    posterior_10k = np.einsum(
        'ij, j...->i...',
        t_base,
        fh['samples'],
    )
    knots_10k = fh['knots'][3:-3]

ps_10k = get_ps_from_samples(posterior_10k, include_mean=False)
prior_ps_10k = get_ps_from_samples(prior_10k, include_mean=False)
ax.plot(
    ls,
    ps_10k,
    # marker='o',
    color='C0',
    label='ARCH10k.1$^*$',
    alpha=1,
)
ax.plot(
    ls[1:],
    prior_ps_10k[1:],
    # marker='o',
    color='C0',
    alpha=0.3,
)


with np.load('../out/covarch_samples.npz') as fh:
    knots_cov = fh['knots']

    prior_cov = fh['prior']
    prior_cov = prior_cov.transpose(2, 1, 0)
    prior_cov = prior_cov[:, :prior_cov.shape[1] // 2, :]

    posterior_cov = fh['samples']
    posterior_cov = posterior_cov.transpose(2, 1, 0)
    posterior_cov = posterior_cov[:, :posterior_cov.shape[1] // 2, :]

ps_cov = get_ps_from_samples(posterior_cov, include_mean=False)
prior_ps_cov = get_ps_from_samples(prior_cov, include_mean=False)
ax.plot(
    ls,
    ps_cov,
    # marker='o',
    color='C3',
    label='COV-ARCH$^*$',
    alpha=1,
)
ax.plot(
    ls,
    prior_ps_cov,
    # marker='o',
    color='C3',
    alpha=0.3,
)


with np.load('../out/kalman_samples.npz') as fh:
    knots_akm = fh['knots']

    prior_akm = fh['prior']
    prior_akm = prior_akm.transpose(2, 1, 0)
    prior_akm = prior_akm[:, :prior_akm.shape[1] // 2, :]

    posterior_akm = fh['samples']
    posterior_akm = posterior_akm.transpose(2, 1, 0)
    posterior_akm = posterior_akm[:, :posterior_akm.shape[1] // 2, :]

ps_akm = get_ps_from_samples(posterior_akm, include_mean=False)
prior_ps_akm = get_ps_from_samples(prior_akm, include_mean=False)
ax.plot(
    ls,
    ps_akm,
    # marker='o',
    color='C2',
    label='ArchKalmag14k$^*$',
    alpha=1,
)
ax.plot(
    ls,
    prior_ps_akm,
    # marker='o',
    color='C2',
    alpha=0.3,
)

handles, _ = ax.get_legend_handles_labels()

manual_line = Line2D(
    [0], [0],
    label='IGRF (1900-2020)',
    color='grey',
    alpha=0.5
)
manual_line_2 = Line2D([0], [0], label='Kalmag', color='grey', alpha=0.5)

# handles.extend(
#     [
#         manual_line,
#         # manual_line_2,
#     ]
# )

# ax.set_title('Prior power @ CMB')
ax.set_xticks(ls)
ax.set_yscale('log')
ax.legend(frameon=False, handles=handles)
ax.grid(alpha=0.2)

fig.savefig(
    '../fig/posterior_spectra.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)

plt.close(fig)

fig, ax = plt.subplots(
    1, 1,
    figsize=(0.4 * paperwidth, 0.4 * paperwidth))

ax.set_xlabel(r'degree $\ell$')
ax.set_ylabel(r'r$_\mu$ [$\mu$T$^2$]')

ps_10k_mean = get_ps_from_samples(posterior_10k, include_vars=False)
ax.plot(
    ls,
    ps_10k_mean,
    # marker='o',
    color='C0',
    label='ARCH10k.1$^*$',
    alpha=1,
)

ps_cov_mean = get_ps_from_samples(posterior_cov, include_vars=False)
ax.plot(
    ls,
    ps_cov_mean,
    # marker='o',
    color='C3',
    label='COV-ARCH$^*$',
    alpha=1,
)

ps_akm_mean = get_ps_from_samples(posterior_akm, include_vars=False)
ax.plot(
    ls,
    ps_akm_mean,
    # marker='o',
    label='ArchKalmag14k$^*$',
    color='C2',
    alpha=1,
)

ax.set_xticks(ls)
ax.set_yscale('log')
ax.legend(frameon=False, handles=handles)
ax.grid(alpha=0.2)

fig.savefig(
    '../fig/posterior_mean_spectra.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)
