import copy

import numpy as np

import torch

from paleokalmag import ChunkedData, CoreFieldModel, settings

from read_ERDA2101 import tab as raw_data

settings.trunc_dir = 0
settings.trunc_int = 0

pars = torch.from_numpy(
    np.array(
        [
            -38,
            39,
            171,
            118,
            380,
        ]
    ),
).to(
    device=settings.device,
    dtype=settings.dtype,
)

myModel = CoreFieldModel(
    lmax=10,
    R=2800,
    gamma=pars[0],
    alpha_dip=pars[1],
    tau_dip=pars[2],
    alpha_wodip=pars[3],
    tau_wodip=pars[4],
    rho=2.25,
)
if __name__ == '__main__':
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
        '../out/kalman_samples.npz',
        knots=myModel.knots,
        samples=samples,
        prior=prior_samples,
        mean=myModel.mean,
        res_D=res_D,
        res_I=res_I,
        res_F=res_F,
    )
