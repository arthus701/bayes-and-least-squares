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
ax.set_ylabel(r'r [$\mu$T$^2$]')

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

ax.plot(
    ls,
    ref_powers,
    color='grey',
    zorder=0,
    alpha=0.5,
)

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
    knots_10k = fh['knots'][3:-3]


ps_10k = get_ps_from_samples(prior_10k)

ax.plot(
    ls[1:],
    ps_10k[1:],
    # marker='o',
    color='C0',
    label='ARCH10k.1',
    alpha=1,
)

with np.load('../out/covarch_samples.npz') as fh:
    knots_cov = fh['knots']
    prior_cov = fh['prior']
    prior_cov = prior_cov.transpose(2, 1, 0)
    prior_cov = prior_cov[:, :prior_cov.shape[1] // 2, :]


ps_cov = get_ps_from_samples(prior_cov)

ax.plot(
    ls,
    ps_cov,
    # marker='o',
    color='C3',
    label='COV-ARCH',
    alpha=1,
)
with np.load('../out/kalman_samples.npz') as fh:
    knots_akm = fh['knots']
    prior_akm = fh['prior']
    prior_akm = prior_akm.transpose(2, 1, 0)
    prior_akm = prior_akm[:, :prior_akm.shape[1] // 2, :]

ps_akm = get_ps_from_samples(prior_akm)

ax.plot(
    ls,
    ps_akm,
    # marker='o',
    color='C2',
    label='ArchKalmag14k',
    alpha=1,
)

handles, _ = ax.get_legend_handles_labels()

manual_line = Line2D(
    [0], [0],
    label='IGRF (1900-2020)',
    color='grey',
    alpha=0.5
)
manual_line_2 = Line2D([0], [0], label='Kalmag', color='grey', alpha=0.5)

handles.extend(
    [
        manual_line,
        # manual_line_2,
    ]
)

# ax.set_title('Prior power @ CMB')
ax.set_xticks(ls)
ax.set_yscale('log')
ax.legend(frameon=False, handles=handles)
ax.grid(alpha=0.2)

fig.savefig(
    '../fig/prior_spectra.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)
