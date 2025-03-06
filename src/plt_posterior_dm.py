import numpy as np

from matplotlib import pyplot as plt

from scipy.interpolate import BSpline

paperwidth = 46.9549534722518 / 4

fac = 0.63712**3

with np.load('../out/arch10k_samples.npz') as fh:
    t_base = BSpline.design_matrix(
        fh['knots'][3:-3],
        fh['knots'],
        3
    ).toarray()
    coeffs_10k = np.einsum(
        'ij, j...->i...',
        t_base,
        fh['samples'],
    )
    knots_10k = fh['knots'][3:-3]

    dm_10k = fac * np.sqrt((coeffs_10k[:, :3]**2).sum(axis=1))

with np.load('../out/kalman_samples.npz') as fh:
    knots_km = fh['knots']
    coeffs_km = fh['samples']
    coeffs_km = coeffs_km.transpose(2, 1, 0)
    coeffs_km = coeffs_km[:, :coeffs_km.shape[1] // 2, :]

    dm_km = fac * np.sqrt((coeffs_km[:, :3]**2).sum(axis=1))

with np.load('../out/covarch_samples.npz') as fh:
    knots_cov = fh['knots']
    coeffs_cov = fh['samples']
    coeffs_cov = coeffs_cov.transpose(2, 1, 0)
    coeffs_cov = coeffs_cov[:, :coeffs_cov.shape[1] // 2, :]

    dm_cov = fac * np.sqrt((coeffs_cov[:, :3]**2).sum(axis=1))


fig, axs = plt.subplots(
    4, 1,
    figsize=(paperwidth, 0.5*paperwidth),
    sharex=True,
)

axs[1].sharey(axs[0])
axs[2].sharey(axs[0])
axs[0].plot(
    knots_10k,
    dm_10k[:, ::5],
    alpha=0.1,
    color='C0',
)
axs[0].plot(
    knots_10k,
    dm_10k.mean(axis=1),
    color='C0',
    label='ARCH10k.1$^*$'
)
axs[3].plot(
    knots_10k,
    dm_10k.std(axis=1),
    color='C0',
)

axs[1].plot(
    knots_km,
    dm_km[:, ::5],
    alpha=0.1,
    color='C2',
)
axs[1].plot(
    knots_km,
    dm_km.mean(axis=1),
    color='C2',
    label='ArchKalmag14k$^*$'
)
axs[3].plot(
    knots_km,
    dm_km.std(axis=1),
    color='C2',
)

axs[2].plot(
    knots_cov,
    dm_cov[:, ::5],
    alpha=0.1,
    color='C3',
)
axs[2].plot(
    knots_cov,
    dm_cov.mean(axis=1),
    color='C3',
    label='COV-ARCH$^*$'
)
axs[3].plot(
    knots_cov,
    dm_cov.std(axis=1),
    color='C3',
)

for ax in axs[:3]:
    ax.set_ylabel(f'DM [$10^{22}$Am$^2$]')
    ax.legend(frameon=False, loc='upper right')

axs[3].set_ylabel('$\sigma_\mathrm{DM}$ [$10^{22}$Am$^2$]')

axs[0].set_xlim([-8000, 2000])
axs[3].set_xlabel('time [yrs.]')

axs[0].set_ylim([5.6, 12.4])
# axs[1].set_ylim([-10, 10])
# axs[1].set_ylim([-5, 5])

# axs[0].set_yticks([-40, -35, -30, -25, -20])
# axs[1].set_yticks(np.arange(-8, 9, 4))
# axs[1].set_yticks(np.arange(-4, 5, 2))


fig.tight_layout()
fig.align_ylabels()
fig.subplots_adjust(hspace=0.1)

fig.savefig(
    '../fig/posterior_dm.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)

# plt.show()
