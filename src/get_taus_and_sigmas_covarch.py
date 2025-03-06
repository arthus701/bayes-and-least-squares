import numpy as np

from pymagglobal.utils import lmax2N

from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.data_utils import mjd2000

chaos = load_CHAOS_matfile('../dat/chaos6/data/CHAOS-6-x9.mat')
chaos_2005 = chaos.synth_coeffs_tdep(mjd2000(2005))
sv_chaos_2005 = chaos.synth_coeffs_tdep(mjd2000(2005), deriv=1)


def get_sigma(ell, coeffs):
    if ell == 1:
        return np.sqrt(np.sum(coeffs[1:3]**2) / 2)
    sigmas = coeffs[lmax2N(int(ell)-1):lmax2N(int(ell))]

    return np.sqrt(np.sum(sigmas**2) / (2*ell + 1))


def get_tau(ell, coeffs, sv_coeffs, model='chaos'):
    sigma = get_sigma(ell, coeffs)
    sigma_dot = get_sigma(ell, coeffs=sv_coeffs)

    return sigma / sigma_dot


lmax = 10
ells = np.arange(lmax) + 1

for ell in ells:
    print(
        f'{ell:>2d}  '
        f'{get_sigma(int(ell), chaos_2005):>10.4f}  '
        f'{get_tau(ell, chaos_2005, sv_chaos_2005):>10.4f}'
    )
