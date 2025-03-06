import numpy as np

from matplotlib import pyplot as plt


def psd_matern(f, w=1):
    return (4 * w**3) / (w**2 + (2*np.pi*f)**2)**2


def psd_spline(f, w=1):
    return (8 * w**3) / (4 * w**4 + (2 * np.pi * f)**4)


paperwidth = 46.9549534722518 / 4

fig, ax = plt.subplots(
    1, 1,
    figsize=(0.4*paperwidth, 0.4 * paperwidth))

ax.set_xlabel('time [yrs.]')
ax.set_yticks([])

w_cov = 1 / 175.0
w_10k = 1 / 172.76
w_akm = 1 / 190

f = 1 / np.logspace(0, 6, 201)
ax.plot(
    1 / f,
    psd_spline(f, w=w_10k),
    label='ARCH10k.1',
    color='C0',
)
ax.plot(
    1 / f,
    psd_matern(f, w=w_cov),
    label='COV-ARCH',
    color='C3',
)
ax.plot(
    1 / f,
    psd_matern(f, w=w_akm),
    label='ArchKalmag14k',
    color='C2',
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.invert_xaxis()
ax.legend(frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(
    '../fig/prior_psds.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)
