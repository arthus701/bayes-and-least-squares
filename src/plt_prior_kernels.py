import numpy as np

from scipy.linalg import cholesky_banded, cho_solve_banded

from matplotlib import pyplot as plt

from pymagglobal.utils import lmax2N
from pymaginverse import FieldInversion, InputData

from invert_kalman import myModel as ArchKalmag14k
from invert_kalman_COVARCH import COVARCH

from read_arch10k_data import tab as raw_data

paperwidth = 46.9549534722518 / 4

ind = 3
t_min = -2000
t_max = 2000
step = 20
p = 3
l_max = 10
n_samps = 100
n_c = lmax2N(l_max)

# Set up data
raw_data = raw_data.query(f'{t_min} <= t <= {t_max}')
raw_data.reset_index(inplace=True, drop=True)

inv = FieldInversion(
    t_min=t_min,
    t_max=t_max,
    t_step=step,
    maxdegree=l_max,
    verbose=False,
)
t = inv.knots[2:-2]
inpData = InputData(raw_data)
inv.prepare_inversion(inpData, spat_type='ohmic_heating', temp_type='min_acc')

fig, ax = plt.subplots(
    1, 1,
    figsize=(0.4*paperwidth, 0.4 * paperwidth))

ax.set_xlabel('time [yrs.]')
ax.set_yticks([])

# ARCH10k.1

lambda_s = 0.7e-13
lambda_t = 0.5e-1

lambda_s *= 1e6
lambda_t *= 1e6 / 4 / np.pi
C_m_inv = (
    lambda_s * inv.sdamp_diag
    + lambda_t * inv.tdamp_diag
)

L_m_inv = cholesky_banded(C_m_inv)

rhs = np.zeros(
    (inv.nr_splines * n_c, inv.nr_splines)
)
for it in range(inv.nr_splines):
    rhs[ind + it*n_c, it] = 1
arch10k = cho_solve_banded(
    (L_m_inv, False),
    rhs,
)
arch10k = arch10k.reshape(inv.nr_splines, n_c, inv.nr_splines)
arch10k = arch10k[inv.nr_splines//2, ind, :]
arch10k /= arch10k.max()

# ArchKalmag14k
archkalmag = np.array(
    [
        ArchKalmag14k.F(dt)[ind, ind] for dt in t
    ]
)
# COV-Arch
covModel = COVARCH()
covarch = np.array(
    [
        covModel.F(dt)[ind, ind] for dt in t
    ]
)

ax.plot(
    t,
    arch10k,
    label='ARCH10k.1',
    color='C0',
)
ax.plot(
    t,
    archkalmag,
    label='ArchKalmag14k',
    color='C2',
)
ax.plot(
    t,
    covarch,
    label='COV-ARCH',
    color='C3',
)
ax.legend(frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(
    '../fig/prior_kernels.pdf',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.,
)
