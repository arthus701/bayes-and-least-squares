import numpy as np

paperwidth = 46.9549534722518 / 4


def kernel(x, y=None, alpha=1., beta=1.):
    if y is None:
        y = x
    x = np.asarray(np.atleast_1d(x))
    y = np.asarray(np.atleast_1d(y))

    # sigma = np.sqrt(2) / 4 / beta**(1 / 4) / alpha**(3 / 4)
    sigma = 1 / beta**(1 / 4) / alpha**(3 / 4)
    tau = np.sqrt(2) * (beta / alpha)**(1 / 4)

    frac = np.abs(x[:, None] - y[None, :]) / tau
    return 1 / 2 * sigma * np.exp(-frac) * np.sin(frac + np.pi / 4)


def dkernel(x, y=None, alpha=1., beta=1.):
    if y is None:
        y = x
    x = np.asarray(np.atleast_1d(x))
    y = np.asarray(np.atleast_1d(y))

    # sigma = np.sqrt(2) / 4 / beta**(1 / 4) / alpha**(3 / 4)
    sigma = 1 / beta**(1 / 4) / alpha**(3 / 4)
    tau = np.sqrt(2) * (beta / alpha)**(1 / 4)

    frac = np.abs(x[:, None] - y[None, :]) / tau

    return (
        -np.sign(x[:, None] - y[None, :]) / 2
        * sigma / tau * np.exp(-frac)
        * (np.sin(frac + np.pi / 4) - np.cos(frac + np.pi / 4))
    )


def ddkernel(x, y=None, alpha=1., beta=1.):
    if y is None:
        y = x
    x = np.asarray(np.atleast_1d(x))
    y = np.asarray(np.atleast_1d(y))

    # sigma = np.sqrt(2) / 4 / beta**(1 / 4) / alpha**(3 / 4)
    sigma = 1 / beta**(1 / 4) / alpha**(3 / 4)
    tau = np.sqrt(2) * (beta / alpha)**(1 / 4)

    frac = np.abs(x[:, None] - y[None, :]) / tau
    return -sigma / tau**2 * np.exp(-frac) * np.cos(frac + np.pi / 4)


def matern32(x, y=None, tau=3.7, sigma=1.):
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)

    frac = (x[:, None] - y[None, :]) / tau
    res = sigma**2 * (1 + np.abs(frac)) * np.exp(-np.abs(frac))

    return res.reshape(x.shape[0], y.shape[0])


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    arr, dt = np.linspace(-10, 10, 2001, retstep=True)
    k = kernel(arr, 0, beta=2, alpha=4)
    print(sum(k) * dt)
    plt.plot(arr, k)
    plt.show()
