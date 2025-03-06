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

with np.load('../out/covarch_samples.npz') as fh:
    knots_cov = fh['knots']
    coeffs_cov = fh['samples']
    coeffs_cov = coeffs_cov.transpose(2, 1, 0)
    coeffs_cov = coeffs_cov[:, :coeffs_cov.shape[1] // 2, :]

n = 4000
grid = get_grid(
    n,
    # R=3480,
)
grid[1] -= 180

basis = dsh_basis(10, grid)

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
# fig_2.subplots_adjust(hspace=-0.5)

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


epoch = -1000
f_levels = np.linspace(0, 12, 17)
z_levels = np.linspace(35, 435, 9)

# epoch = -3650
# f_levels = np.linspace(0, 16, 9)
# z_levels = np.linspace(35, 435, 9)

f_levels_norm = np.linspace(0.04, 1, 17)
z_levels_norm = np.linspace(0.44, 1, 9)

for it, (knots, _coeffs, label) in enumerate(
    zip(
        [
            knots_10k,
            knots_km,
            knots_cov,
        ],
        [
            coeffs_10k,
            coeffs_km,
            coeffs_cov,
        ],
        [
            'ARCH10k.1$^*$',
            'ArchKalmag14k$^*$',
            'COV-ARCH$^*$',
        ],
    )
):
    t_ind = np.argmin(abs(knots - epoch))
    if 0 < abs(knots[t_ind] - epoch):
        print(
            f"Epoch {epoch} could not be matched exactly, "
            f"using model {label} at {knots[t_ind]} instead."
        )

    coeffs = _coeffs[t_ind, :, :]

    intensity = np.einsum(
        'i..., ijk -> jk...',
        coeffs,
        basis.reshape(120, -1, 3),
    ).T
    intensity = np.sqrt(np.sum(intensity**2, axis=1)).std(axis=0)

    f_surf = axs_2[0, it].tricontourf(
        plt_lat, plt_lon, intensity,
        cmap='magma_r', zorder=0,
        levels=f_levels,
        extend='max',
    )

    f_rel = axs_2[1, it].tricontourf(
        plt_lat, plt_lon, intensity / intensity.max(),
        cmap='magma_r', zorder=0,
        levels=f_levels_norm,
    )

    axs_2[0, it].set_title(label)


int_cbar = fig_2.colorbar(
    f_surf,
    cax=colaxs[0],
    orientation='vertical',
)
int_cbar.ax.set_ylabel(r'$\sigma_F$ [$\mu$T]')
int_cbar.ax.set_yticks(f_levels[::2])
int_cbar.ax.set_yticklabels(f_levels[::2])
int_cbar.ax.tick_params(labelsize=8)

int_cbar_rel = fig_2.colorbar(
    f_rel,
    cax=colaxs[1],
    orientation='vertical',
)
int_cbar_rel.ax.set_ylabel(r'$\sigma_F$ [norm.]')
int_cbar_rel.ax.set_yticks(f_levels_norm[::2])
int_cbar_rel.ax.set_yticklabels(
    [f'{val:.2f}' for val in f_levels_norm[::2]])
int_cbar_rel.ax.tick_params(labelsize=8)

fig_2.suptitle(f'{abs(epoch):d} BCE', fontsize=14, y=1.03)

fig_2.savefig(
    f'../fig/posterior_std_s_{abs(epoch):d}-BCE.pdf',
    bbox_inches='tight',
    pad_inches=0.,
    transparent=True,
)

# plt.show()
