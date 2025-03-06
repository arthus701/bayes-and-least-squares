import numpy as np

import torch

from pymagglobal.utils import i2lm_l, lmax2N, scaling

from paleokalmag import settings, CoreFieldModel
from paleokalmag.kalman import Kalman
from paleokalmag.utils import REARTH


class ADP2Model(CoreFieldModel):
    def __init__(self, lmax, R, gamma, alpha_dip,
                 log_omega, log_xi, alpha_wodip, tau_wodip, rho=0.):
        Kalman.__init__(self)
        self.lmax = lmax
        self.R = R
        self.gamma = gamma
        self.alpha_dip = alpha_dip
        self.omega_dip = torch.exp(log_omega)
        self.xi_dip = torch.exp(log_xi)
        self.chi_dip = torch.sqrt(self.xi_dip**2 + self.omega_dip**2)
        self.alpha_wodip = alpha_wodip
        self.tau_wodip = tau_wodip
        self.rho = rho
        self.eps = 1.

        self.n_coeffs = lmax2N(self.lmax)

        scl = torch.from_numpy(
            scaling(self.R, REARTH, self.lmax)**2
        ).to(device=settings.device, dtype=settings.dtype)

        self.diag = self.alpha_wodip**2 * torch.ones(
            self.n_coeffs,
            device=settings.device,
            dtype=settings.dtype,
        )
        self.diag[0] = self.alpha_dip**2
        self.diag *= scl

        ils = np.array(
            [1. / float(i2lm_l(i)) for i in range(self.n_coeffs)],
        )
        ils = torch.from_numpy(ils)
        self.ils = ils.to(settings.device, dtype=settings.dtype)

        self.taus = self.tau_wodip * self.ils
        self.taus[0] = 1 / self.omega_dip
        self.taus[1:3] *= self.ils[3]

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

    def update_F(self, delta_t=50):
        ns = torch.arange(
            self.n_coeffs,
            device=settings.device,
            dtype=int,
        )

        frac = delta_t / self.taus
        exp = torch.exp(-torch.abs(frac))

        F = torch.zeros(
            (2*self.n_coeffs, 2*self.n_coeffs),
            device=settings.device,
            dtype=settings.dtype,
        )

        F[ns, ns] = (1 + torch.abs(frac)) * exp
        F[ns, ns+self.n_coeffs] = delta_t * exp
        F[ns+self.n_coeffs, ns] = -frac / self.taus * exp
        F[ns+self.n_coeffs, ns+self.n_coeffs] = (1 - torch.abs(frac)) * exp

        F[0, 0] = 0.5 / self.xi_dip \
            * ((self.xi_dip + self.chi_dip)
               * torch.exp((self.xi_dip - self.chi_dip)*abs(delta_t))
               + (self.xi_dip - self.chi_dip)
               * torch.exp(-(self.xi_dip + self.chi_dip)*abs(delta_t)))
        F_01 = torch.exp(-self.chi_dip*abs(delta_t)) \
            * torch.sinh(self.xi_dip*delta_t) / self.xi_dip
        F[0, self.n_coeffs] = F_01
        F[self.n_coeffs, 0] = -self.omega_dip**2 * F_01
        F[self.n_coeffs, self.n_coeffs] = 0.5 / self.xi_dip \
            * ((self.xi_dip - self.chi_dip)
               * torch.exp((self.xi_dip - self.chi_dip)*abs(delta_t))
               + (self.xi_dip + self.chi_dip)
               * torch.exp(-(self.xi_dip + self.chi_dip)*abs(delta_t)))

        self._F = F
