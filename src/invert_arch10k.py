import numpy as np

from scipy.linalg import solve_banded, cholesky_banded

from pymagglobal.utils import lmax2N
from pymaginverse import FieldInversion, InputData
from pymaginverse.forward_modules import (frechet_types,
                                          forward_obs_time)
from pymaginverse.banded_tools.build_banded import build_banded

from read_ERDA2206 import tab as raw_data

t_min = -10010
t_max = 1990
step = 40
p = 3
l_max = 10
n_samps = 100
n_c = lmax2N(l_max)

# Set up data
raw_data = raw_data.query(f'{t_min} <= t <= {t_max}')
raw_data.reset_index(inplace=True, drop=True)

lambda_s = 0.7e-13
lambda_t = 0.5e-1

lambda_s *= 1e6
lambda_t *= 1e6 / 4 / np.pi

rng = np.random.default_rng(seed=12331)


inv = FieldInversion(
    t_min=t_min,
    t_max=t_max,
    t_step=step,
    maxdegree=l_max,
    verbose=False,
)
inpData = InputData(raw_data)

inv.prepare_inversion(inpData, spat_type='ohmic_heating', temp_type='min_acc')
x0 = np.zeros(n_c)
x0[0] = -30
inv.run_inversion(x0, max_iter=10, spat_damp=lambda_s, temp_damp=lambda_t)

C_m_inv = (
    lambda_s * inv.sdamp_diag
    + lambda_t * inv.tdamp_diag
)
C_m_inv[-1] += 1e-15*np.min(np.abs(C_m_inv[-1]))

std_normal = rng.normal(size=(inv.nr_splines*n_c, n_samps))

L_m_inv = cholesky_banded(C_m_inv)
pri_samps = solve_banded((0, L_m_inv.shape[0]-1), L_m_inv, std_normal)
pri_samps = pri_samps.reshape(inv.nr_splines, n_c, n_samps)

rng = np.random.default_rng(161)

forwobs_matrix = forward_obs_time(
    inv.coeffs_solution,
    inv.spatial,
    inv.temporal,
)
frech_matrix = frechet_types(
    inv.spatial, forwobs_matrix
)
frech_matrix = np.vstack(
    (
        frech_matrix[inv.idx_res[0]:inv.idx_res[1], 0],
        frech_matrix[inv.idx_res[1]:inv.idx_res[2], 1],
        frech_matrix[inv.idx_res[2]:inv.idx_res[3], 2],
        frech_matrix[inv.idx_res[3]:inv.idx_res[4], 3],
        frech_matrix[inv.idx_res[4]:inv.idx_res[5], 4],
        frech_matrix[inv.idx_res[5]:inv.idx_res[6], 5],
        frech_matrix[inv.idx_res[6]:inv.idx_res[7], 6],
    )
).T
# include the C_e^{-1/2} factor
frech_matrix /= inv.std[None, :]
pst_prc = build_banded(
    np.ascontiguousarray(frech_matrix),
    inv.temporal,
    inv._SPL_DEGREE,
    inv.ind_list,
    inv.starts,
)
pst_prc[pst_prc.shape[0]-C_m_inv.shape[0]:] += C_m_inv

std_normal = rng.normal(size=(inv.nr_splines*n_c, n_samps))

L_pst_inv = cholesky_banded(pst_prc)
pst_samps = solve_banded((0, L_pst_inv.shape[0]-1), L_pst_inv, std_normal)
pst_samps += inv.coeffs_solution.flatten()[:, None]
pst_samps = pst_samps.reshape(inv.nr_splines, n_c, n_samps)

np.savez(
    '../out/arch10k_samples.npz',
    knots=inv.knots,
    prior=pri_samps,
    samples=pst_samps,
)
