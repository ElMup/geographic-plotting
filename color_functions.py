from matplotlib import colors, cm
import numpy as np


def zero_different_colormap(
    vmax: int,
    cmap: str = "viridis",
    zero_color: np.array = np.array([1, 1, 1, 1]),
    stepsize=1,
):
    """Generate a colorlist in which zero gets another color. Useful for plotting discrete data where u want 0 to have
    a different representation

    Parameters
    ----------
    vmax
    cmap
    zero_color
    stepsize

    Returns
    -------

    """
    # subtract one for zero value color
    vmax = vmax * stepsize - 1

    # make colormap
    colorlist = list(
        cm.get_cmap(cmap)(
            np.linspace(
                0,
                1,
                vmax,
            )
        )
    )
    colorlist.insert(0, zero_color)

    #: colormap for continuous raster layer
    return colors.ListedColormap(colorlist)
