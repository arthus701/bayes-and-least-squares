import copy

import numpy as np

import torch

from pymagglobal.utils import REARTH, lmax2N, i2lm_l
from paleokalmag import ChunkedData, settings
from adp_2_model import ADP2Model

from paleokalmag.kalman import Kalman

from read_ERDA2206 import tab as raw_data


class COVARCH(ADP2Model):
    def __init__(self):
        Kalman.__init__(self)
        self.lmax = 10
        self.R = REARTH
        self.gamma = -23
        self.sigma_dip = 6
        self.omega_dip = 1/400 * torch.ones(1, device=settings.device)
        self.xi_dip = 1/20 * torch.ones(1, device=settings.device)
        self.chi_dip = torch.sqrt(self.xi_dip**2 + self.omega_dip**2)
        self.rho = 0.
        self.eps = 1.

        self.n_coeffs = lmax2N(self.lmax)

        _sigmas = [
            3779.6933,
            2213.3759,
            1164.7702,
            462.7909,
            172.8008,
            68.5210,
            35.9837,
            13.3939,
            9.0052,
            3.4806,
        ]

        _taus = [
            195.9,
            175.0,
            223.9,
            116.0,
            94.8,
            63.7,
            66.3,
            40.5,
            44.2,
            31.2,
        ]

        alphas = np.array(
            [
                _sigmas[i2lm_l(it)-1]**2 * 1e-6 for it in range(self.n_coeffs)
            ]
        )
        taus = np.array([_taus[i2lm_l(it)-1] for it in range(self.n_coeffs)])
        alphas[0] = self.sigma_dip**2
        taus[0] = 1 / self.omega_dip

        diag = torch.from_numpy(
            alphas
        )
        self.diag = diag.to(settings.device, dtype=settings.dtype)

        taus = torch.from_numpy(
            taus,
        )
        self.taus = taus.to(settings.device, dtype=settings.dtype)

        self._cov_inf = torch.diag(
            torch.concatenate(
                (
                    self.diag,
                    self.diag/self.taus**2,
                )
            )
        )
        self._prior_mean = torch.zeros(
            2*self.n_coeffs,
            device=settings.device,
            dtype=settings.dtype,
        )
        self._prior_mean[0] = self.gamma


if __name__ == '__main__':
    myModel = COVARCH()

    resModel = copy.deepcopy(myModel)

    cDat = ChunkedData(
        raw_data, delta_t=10, start=2000, end=-10000,
    )

    myModel.forward(
        cDat,
        output_every=5,
        quiet=False,
    )

    samples = myModel.sample(n_samps=100, quiet=False)

    prior_samples = myModel.sample_prior(
        myModel.knots,
        n_samps=100,
        quiet=False,
    )

    res_D, res_I, res_F = resModel.calc_residuals(
        cDat,
        quiet=False,
    )

    np.savez(
        '../out/covarch_samples.npz',
        knots=myModel.knots,
        samples=samples,
        prior=prior_samples,
        mean=myModel.mean,
        res_D=res_D,
        res_I=res_I,
        res_F=res_F,
    )
