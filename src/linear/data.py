import numpy as np

rng = np.random.default_rng(13311)


def signal(x):
    return 0.1*x + np.sin(x*1.3) + 0.2*np.cos(x*3.51)


n_data = 157
inds = [
    9,
    16,
    7,
]
# inds = np.arange(n_data)
n_points = len(inds)
bound = 4
sigma_o = 0.2
sigma_o_reported = sigma_o

arr = np.linspace(-bound, bound, 401)

x_at = np.unique(bound * (2 * rng.random(n_data) - 1))
y_at = signal(x_at) + rng.normal(scale=sigma_o, size=n_data)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(
        arr,
        signal(arr),
        ls='--',
        color='grey',
        zorder=-1,
    )
    ax.errorbar(
        x_at,
        y_at,
        yerr=sigma_o_reported,
        ls='',
        color='black',
        marker='.',
    )
    fig.tight_layout()

    plt.show()
