This repository collects code for reproducing "The relation between regularized least squares and Bayesian inversion in Geomagnetic field modelling".

The necessary data files can be obtained from [DTU Space](http://www.spacecenter.dk/files/magnetic-models/CHAOS-6/CHAOS-6-x9.mat) and ERDA: https://earthref.org/ERDA/2206/. Place the files `CHAOS-6-x9.mat` and `arch10k.txt` in a folder `dat/` at the root of the repository. A folder `fig` should also be created for figure output.

The required model access and inversion software is publicly available:

* [`chaosmagpy`](https://github.com/ancklo/ChaosMagPy)
* [`paleokalmag`](https://sec23.git-pages.gfz-potsdam.de/korte/paleokalmag/)
* [`pymaginverse`](https://github.com/outfrenk/pymaginverse)


