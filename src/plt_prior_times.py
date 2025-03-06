import numpy as np

from matplotlib import pyplot as plt

from pymagglobal.utils import lm2i

from invert_kalman import myModel as ArchKalmag14k
from invert_kalman_COVARCH import COVARCH


paperwidth = 46.9549534722518 / 4


def tau_spline(ell):
    return (ell * (ell + 1) / ((2 * ell + 1)**2 * (2 * ell + 3)))**(1 / 4)


lambda_s = 0.7e-13
lambda_t = 0.5e-1

# slightly different definition
lambda_s *= 1e6 / (6371.2 / 3485)
lambda_t *= 1e6 / 4 / np.pi
ell = 2
tau_arch = (
    lambda_t / lambda_s
    * ell * (ell + 1)
    / ((2 * ell + 1)**2 * (2 * ell + 3))
) ** (1/4) / np.sqrt(2)

ls = np.arange(10) + 1

taus_arch = tau_spline(ls)
taus_arch *= tau_arch / taus_arch[1]

taus_kalman = ArchKalmag14k.tau_wodip / ls
taus_kalman[0] = ArchKalmag14k.tau_dip

covModel = COVARCH()
taus_cov = [covModel.taus[lm2i(int(ell), 0)] for ell in ls]
taus_cov[0] = covModel.taus[1]

print(f'A10k {taus_arch[1]:.2f}')
print(f'COV {taus_cov[1]:.2f}')
print(f'Akm {taus_kalman[1]:.2f}')
exit()

fig, axs = plt.subplots(
    3, 1,
    sharex=True,
    figsize=(0.4*paperwidth, 0.4 * paperwidth),
)

gs = axs[0].get_gridspec()
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1:])

for ax in axs:
    ax.remove()

ax2.plot(
    ls[1:],
    taus_arch[1:],
    label='ARCH10k.1',
    marker='o',
    color='C0',
)

ax2.plot(
    ls,
    taus_kalman,
    label='ArchKalmag14k',
    marker='o',
    color='C2',
)

ax2.plot(
    ls,
    taus_cov,
    label='COV-ARCH',
    marker='o',
    color='C3',
)
ax1.scatter(
    1,
    1 / covModel.omega_dip,
    marker='d',
    color='C3',
)

handels, labels = ax2.get_legend_handles_labels()
ax1.legend(handels, labels, frameon=False, loc='upper right')
ax1.set_yticks([375, 400, 425])
ax1.set_ylim(360, 440)
xmin, xmax = ax2.get_xlim()
ax1.set_xlim(xmin, xmax)
ax1.set_xticks([])

ax2.set_xlabel(r'degree $\ell$')
ax2.set_xticks(ls)
ax2.set_ylabel(r'$\tau(\ell)$ [yrs.]', y=0.7)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color='k',
    mec='k',
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

fig.tight_layout()
fig.savefig(
    '../fig/prior_times.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)
