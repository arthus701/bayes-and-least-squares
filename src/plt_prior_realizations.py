import numpy as np

from matplotlib import pyplot as plt
from cartopy import crs as ccrs

from pymagglobal.utils import i2lm_l, i2lm_m, get_grid, dsh_basis

from scipy.interpolate import BSpline

paperwidth = 46.9549534722518 / 4


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
    coeffs_10k = np.einsum(
        'ij, j...->i...',
        t_base,
        fh['samples'],
    )
    knots_10k = fh['knots'][3:-3]

with np.load('../out/kalman_samples.npz') as fh:
    knots_km = fh['knots']
    coeffs_km = fh['samples']
    coeffs_km = coeffs_km.transpose(2, 1, 0)
    coeffs_km = coeffs_km[:, :coeffs_km.shape[1] // 2, :]
    prior_km = fh['prior']
    prior_km = prior_km.transpose(2, 1, 0)
    prior_km = prior_km[:, :coeffs_km.shape[1], :]

with np.load('../out/covarch_samples.npz') as fh:
    knots_cov = fh['knots']
    coeffs_cov = fh['samples']
    coeffs_cov = coeffs_cov.transpose(2, 1, 0)
    coeffs_cov = coeffs_cov[:, :coeffs_cov.shape[1] // 2, :]
    prior_cov = fh['prior']
    prior_cov = prior_cov.transpose(2, 1, 0)
    prior_cov = prior_cov[:, :coeffs_cov.shape[1], :]

fig, axs = plt.subplots(
    3, 1,
    figsize=(paperwidth, 0.4*paperwidth),
    sharex=True,
    # sharey=True,
)

ind = 3

axs[0].plot(
    knots_10k,
    prior_10k[:, ind, ::5],
    alpha=0.1,
    color='C0',
)
axs[0].plot(
    knots_10k,
    prior_10k[:, ind, 0],
    color='C0',
    label='ARCH10k.1'
)

axs[1].plot(
    knots_km,
    prior_km[:, ind, ::5],
    alpha=0.1,
    color='C2',
)
axs[1].plot(
    knots_km,
    prior_km[:, ind, 0],
    color='C2',
    label='ArchKalmag14k'
)

axs[2].plot(
    knots_cov,
    prior_cov[:, ind, ::5],
    alpha=0.1,
    color='C3',
)
axs[2].plot(
    knots_cov,
    prior_cov[:, ind, 0],
    color='C3',
    label='COV-ARCH'
)

for ax in axs:
    ax.set_ylabel(rf'$g_{i2lm_l(ind):d}^{i2lm_m(ind):d}$ [$\mu$T]')
    ax.legend(frameon=False, loc='upper right')

axs[0].set_xlim([-8000, 2000])
axs[2].set_xlabel('time [yrs.]')

axs[0].set_ylim([-2.3, 2.3])
axs[0].set_yticks(np.arange(-2, 3, 1))
axs[1].set_ylim([-12, 12])
axs[1].set_yticks([-10, -5, 0, 5, 10])
axs[1].set_ylim([-12, 12])
axs[1].set_yticks([-10, -5, 0, 5, 10])
# axs[2].set_ylim([-3.6, 3.6])
axs[2].set_ylim([-8, 8])
axs[2].set_yticks(np.arange(-7, 8, 3.5))

fig.tight_layout()
fig.subplots_adjust(hspace=0.1)
fig.align_ylabels()

fig.savefig(
    '../fig/prior_samples_t.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)

n = 4000
grid = get_grid(
    n,
    # R=3480,
)
grid[1] -= 180
basis = dsh_basis(10, grid)

grid_cmb = get_grid(
    n,
    R=3480,
)
grid_cmb[1] -= 180
basis_cmb = dsh_basis(10, grid_cmb)
proj = ccrs.Robinson()
plt_lat, plt_lon, _ = proj.transform_points(
    ccrs.Geodetic(),
    grid[1],
    90 - grid[0],
).T

fig_2, axs_2 = plt.subplots(
    2, 3,
    figsize=(paperwidth, 0.4*paperwidth),   
    subplot_kw={'projection': proj},
)

for ax in axs_2.flatten():
    ax.set_global()
    ax.coastlines(lw=0.8, zorder=11)

fig_2.tight_layout()
# fig_2.subplots_adjust(hspace=-0.38, top=1., left=0.09, bottom=0.05)
# fig_2.subplots_adjust(hspace=0.4)

bnds = axs_2[0, 0].get_position().bounds
# spacing between plots and labels
spc = 0.01*bnds[3]
cbar_hght = 0.012

colaxs = np.empty(axs_2.shape[0], dtype='object')
for jt in range(axs_2.shape[0]):
    bnds = axs_2[jt, 2].get_position().bounds
    # colorbar for the mean
    colaxs[jt] = fig_2.add_axes([
        bnds[0]+bnds[2]+spc+1.5*cbar_hght,
        bnds[1],
        cbar_hght,
        bnds[3],
    ])

f_levels = np.linspace(0, 1, 17)
z_levels = np.linspace(-1, 1, 17)

for it, (coeffs, label) in enumerate(
    zip(
        [
            prior_10k[50, :, 0].copy(),
            prior_km[50, :, 0].copy(),
            prior_cov[50, :, 0].copy(),
        ],
        [
            'ARCH10k.1',
            'ArchKalmag14k',
            'COV-ARCH',
        ],
    )
):
    coeffs[0:3] = 0.

    intensity = np.einsum(
        'i, ijk -> jk',
        coeffs,
        basis.reshape(120, -1, 3),
    ).T
    intensity = np.sqrt(np.sum(intensity**2, axis=0))
    intensity /= np.abs(intensity).max()

    f_surf = axs_2[0, it].tricontourf(
        plt_lat, plt_lon, intensity,
        cmap='viridis_r', zorder=0,
        levels=f_levels,
    )

    z_cmb = np.einsum(
        'i, ijk -> jk',
        coeffs,
        basis_cmb.reshape(120, -1, 3),
    ).T[2]
    z_cmb /= np.abs(z_cmb).max()

    br_cmb = axs_2[1, it].tricontourf(
        plt_lat, plt_lon, z_cmb,
        cmap='RdBu', zorder=0,
        levels=z_levels,
    )

    axs_2[0, it].set_title(label)


z_cbar = fig_2.colorbar(
    br_cmb,
    cax=colaxs[1],
    orientation='vertical',
)
z_cbar.ax.set_ylabel(r'$B_Z$ @ CMB [norm.]')
z_cbar.ax.set_yticks(z_levels[::4])
z_cbar.ax.set_yticklabels(z_levels[::4])
z_cbar.ax.tick_params(labelsize=8)

int_cbar = fig_2.colorbar(
    f_surf,
    cax=colaxs[0],
    orientation='vertical',
)
int_cbar.ax.set_ylabel(r'$F$ [norm.]')
int_cbar.ax.set_yticks(f_levels[::4])
int_cbar.ax.set_yticklabels(f_levels[::4])
int_cbar.ax.tick_params(labelsize=8)

fig_2.align_ylabels()
fig_2.savefig(
    '../fig/prior_samples_s.pdf',
    bbox_inches='tight',
    pad_inches=0.,
    transparent=True,
)


# plt.show()
