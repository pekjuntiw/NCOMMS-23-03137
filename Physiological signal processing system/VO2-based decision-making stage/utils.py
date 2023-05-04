import numpy as np
import matplotlib.pyplot as plt
import torch


def absId(kappa, w_over_l, vgs: torch.Tensor, vth, vds: torch.Tensor):
    """
        MOSFET drain current
    """
    sat = 0.5 * kappa * w_over_l * (vgs - vth) ** 2
    lin = 0.5 * kappa * w_over_l * (2 * (vgs - vth) - vds) * vds
    Ids = torch.where(vds >= vgs - vth, sat, lin) * torch.where(vgs > vth, 1, 0)

    return Ids


def raster_plot(ts, sp_matrix, ax=None, marker='.', markersize=2, color='k', xlabel='Time (ms)', ylabel='Neuron index',
                xlim=None, ylim=None, title=None, show=False, **kwargs):
    """
        Raster plot to visualize spikes easily
    """

    sp_matrix = np.asarray(sp_matrix)
    if ts is None:
        raise
    ts = np.asarray(ts)

    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    time = ts[elements[0]]

    # plot raster
    if ax is None:
        ax = plt
    ax.plot(time, index, marker + color, markersize=markersize, **kwargs)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    if show:
        plt.show()
