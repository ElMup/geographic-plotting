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


def get_rgba(seed: int, alpha: float) -> list[float, float, float, float]:
    """Generate three floats between 0 and 1 for use as rgba color.

    :param seed:
    :param alpha:
    :return:
    """
    np.random.seed(seed)
    return [
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        alpha,
    ]


def darken_colorlist(colorlist) -> list:
    darken_factor = 0.7
    darkened_colorlist = [
        [
            color[0] * darken_factor,
            color[1] * darken_factor,
            color[2] * darken_factor,
            color[3],
        ]
        for color in colorlist
    ]
    return darkened_colorlist
