from abc import abstractmethod, ABC
from collections.abc import Iterable

import contextily as cx
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from geopandas import GeoDataFrame
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.plot import plotting_extent

from .color_functions import get_rgba, darken_colorlist

from dataclasses import dataclass, field


@dataclass
class LegendSettings:
    title: str
    inset: bool = True
    lax_kwargs: dict = field(default_factory=dict)
    legend_kwargs: dict = field(default_factory=dict)
    colorbar_kwargs: dict = field(default_factory=lambda: {"aspect": 25})
    colorbar_label_fontsize: int = 12
    n_ticks: int = 5


class Layer(ABC):
    @abstractmethod
    def draw(self, ax: plt.Axes):
        """Draw this layer upon axes

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to draw this layer on
        """
        pass


class RasterLayer(Layer):
    def __init__(self, raster, transform):
        self.raster = raster
        self.transform = transform
        self.cmap = "viridis"

        #: ceiling of raster values
        self.ceiling = np.ceil(np.nanmax(self.raster))
        if np.isnan(self.ceiling):
            self.ceiling = 1

        # bottom of raster values
        self.floor = np.floor(np.nanmin(self.raster))

    def draw(self, ax: plt.Axes):
        """Draw management type map as background

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to draw layer on
        """
        cmap = cm.get_cmap(self.cmap)
        cmap.set_bad(alpha=0)
        # Draw actual raster layer
        ax.imshow(
            self.raster,
            cmap=cmap,
            norm=self.norm,
            extent=plotting_extent(self.raster, self.transform),
            interpolation="nearest",
        )


class ContinuousRasterLayer(RasterLayer):
    """Raster layer with continuous data

    Parameters
    ----------
    raster : numpy.array
        data as numpy array
    transform : rasterio.transform
        rasterio transform object to know where to plot layer
    cmap : str, matplotlib.colors.ListedColormap
        string or ListedColormap specifying which colormap to plot with
    stepsize : int
        the number of steps between each whole number. default is 10, so that the colormap is divided into steps of 0.1
    """

    def __init__(
        self,
        raster: np.array,
        transform: rio.transform,
        cmap: str or list = "viridis",
        stepsize: int = 10,
    ):
        super().__init__(raster, transform)

        #: colormap for continuous raster layer
        self.cmap = matplotlib.colormaps[cmap]
        self.stepsize = stepsize

        #: make normalization
        self.norm = colors.BoundaryNorm(
            boundaries=np.arange(
                self.floor, self.ceiling + self.stepsize, self.stepsize
            ),
            ncolors=self.cmap.N,
        )


class CategoricalRasterLayer(RasterLayer):
    """Raster layer with categorical data as integers

    Parameters
    ----------
    raster : numpy.array
        data as numpy array of integers
    transform : rasterio.transform
        rasterio transform object to know where to plot layer
    colorlist : list
        list of colors as tuples of values between 0 and 1: (r, g, b, alpha)
    """

    def __init__(self, raster: np.array, transform: rio.transform, colorlist: list):
        super().__init__(raster, transform)

        #: normalization for layer
        self.norm = colors.BoundaryNorm(
            boundaries=np.arange(0, self.ceiling) + 0.5, ncolors=self.ceiling
        )

        #: colormap for this layer
        self.cmap = colors.ListedColormap(colorlist)


class VectorLayer(Layer):
    """Vector layer

    Parameters
    ----------
    vector: geopandas.GeoDataFrame
        GeodataFrame containing the _vector
    kwargs
        Keyword arguments to set layer properties
    """

    def __init__(self, vector: GeoDataFrame, **kwargs):
        self._vector = vector
        self.n_shapes = len(vector)
        self.facecolor = "none"
        self._kwargs = kwargs
        self._label_settings = None

    def draw(self, ax, **kwargs):
        """Draw this _vector layer on the Axes

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to plot everything upon

        """
        self._vector.plot(ax=ax, aspect=1, **self._kwargs)

        if self._label_settings is not None:
            for setting in self._label_settings:
                ax.annotate(**setting)

    def add_shape_labels(self, column: str, **kwargs):
        """Automatically add a label for every polygon in this layer. You can pass kwargs used for annotate, like path
        effects. kwargs of single length will be made to lists of same length as N labels.

        Parameters
        ----------
        column: str
            The column in VectorLayer dataframe to use for shape label text
        kwargs

        Returns
        -------

        """
        self._label_settings = []
        for i, row in self._vector.reset_index().iterrows():
            settings = {}

            # Add text
            if isinstance(row[column], float):
                settings["text"] = f"{row[column]:.2f}"
            else:
                settings["text"] = row[column]

            # Add coordinates
            if "xy" in kwargs:
                settings["xy"] = kwargs["xy"][i]
            else:
                settings["xy"] = row.geometry.representative_point().coords[0]

            # Add each keyword argument
            for kw, arg in kwargs.items():
                if kw not in ["text", "xy"]:
                    if not isinstance(arg, str) and isinstance(arg, Iterable):
                        settings[kw] = arg[i]
                    else:
                        settings[kw] = arg
            self._label_settings.append(settings)


class NumericVectorLayer(VectorLayer):
    def __init__(self, vector: GeoDataFrame, cmap: str = "viridis", **kwargs):
        if "color" not in kwargs or kwargs["color"] is None:
            self.cmap = matplotlib.colormaps[cmap]
            kwargs["color"] = [self.cmap(self.norm(value)) for value in vector.index]
        else:
            self.cmap = colors.LinearSegmentedColormap.from_list(
                "Temporary_cmap", kwargs["color"]
            )

        # add darkened edgecolors
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = darken_colorlist(kwargs["color"])
        super().__init__(vector, **kwargs)


class ContinuousVectorLayer(NumericVectorLayer):
    def __init__(
        self,
        vector: GeoDataFrame,
        stepsize: int = 1,
        **kwargs,
    ):
        self.stepsize = stepsize
        self.ceiling = np.ceil(vector.index.max())
        self.floor = np.floor(vector.index.min())
        self.norm = colors.BoundaryNorm(
            np.arange(self.floor, self.ceiling, self.stepsize), 256
        )
        super().__init__(vector, **kwargs)


class OrdinalVectorLayer(NumericVectorLayer):
    def __init__(
        self,
        vector: GeoDataFrame,
        **kwargs,
    ):
        if vector.index.nunique() == 1:
            raise ValueError(
                "Not enough categories, provide a GeodataFrame column with more than 1 category"
            )
        else:
            self.stepsize = 1
            self.categories = vector.index.unique()
            self.ceiling = max(self.categories)
            self.floor = min(self.categories)
            self.norm = matplotlib.colors.Normalize(vmin=self.floor, vmax=self.ceiling)
            super().__init__(vector, **kwargs)


class CategoricalVectorLayer(VectorLayer):
    def __init__(self, vector: GeoDataFrame, **kwargs):
        # save categories in class as they are needed when making a legend
        self.categories = vector.index.unique()

        # make normalization
        self.norm = {category: i for i, category in enumerate(self.categories)}.get

        # set colors if not given
        if "color" not in kwargs or kwargs["color"] is None:
            kwargs["color"] = [get_rgba(self.norm(value), 1) for value in vector.index]
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = darken_colorlist(kwargs["color"])

        #: colormap for this layer
        self.cmap = colors.ListedColormap(kwargs["color"])

        super().__init__(vector, **kwargs)


class Legend(ABC):
    def __init__(self, layer, settings):
        self.layer = layer
        self.settings = settings

    @abstractmethod
    def draw(self, lax: Axes):
        pass

    @abstractmethod
    def _get_lax(self, **kwargs):
        pass


class MtLegend(Legend):
    """Management type legend class

    Parameters
    ----------
    categorical_raster : CategoricalRasterLayer
        Management type layer with management types as integers
    dictionary : dict
        Dictionary to translate integer values to labels or landscape type codes
    """

    def __init__(
        self,
        categorical_layer: CategoricalRasterLayer
        | CategoricalVectorLayer
        | OrdinalVectorLayer,
        settings: LegendSettings,
    ):
        super().__init__(categorical_layer, settings)

    def _get_lax(self, ax: Axes, **kwargs):
        if self.settings.inset:
            lax = inset_axes(ax, **self.settings.lax_kwargs)
        else:
            lax = make_axes_locatable(ax).append_axes(**self.settings.lax_kwargs)
        lax.axis("off")
        return lax

    def draw(self, ax):
        """

        Parameters
        ----------
        divider
            AxesDivider which will make seperate axes for management type legend. `Example for how to make divider <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>`_


        """

        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(
                color=self.layer.cmap(self.layer.norm(category)),
                label=category,
            )
            for category in self.layer.categories
        ]

        # put those patched as legend-handles into the legend
        legend = self._get_lax(ax).legend(
            handles=patches,
            title=self.settings.title,
            **self.settings.legend_kwargs,
        )
        legend.get_frame().set_linewidth(0.8)


class ColorBar(Legend):
    """A color bar, pretty straightforward

    Parameters
    ----------
    continuous_layer
        Raster layer to take normalization and colormap from
    label: str
        Label vor the color bar
    """

    def _get_lax(self, ax: Axes, **kwargs):
        if self.settings.inset:
            return inset_axes(ax, **self.settings.lax_kwargs)
        else:
            return make_axes_locatable(ax).append_axes(**self.settings.lax_kwargs)

    def __init__(
        self,
        layer: ContinuousRasterLayer | ContinuousVectorLayer,
        settings: LegendSettings,
    ):
        super().__init__(layer, settings)
        self.mappable = cm.ScalarMappable(layer.norm, layer.cmap)

        if -1 <= layer.ceiling <= 1:
            ticks_step = 0.1
        else:
            ticks_step = (layer.ceiling - layer.floor) / settings.n_ticks
            if ticks_step >= 1:
                ticks_step = np.floor(ticks_step)

        self.ticks = np.arange(layer.floor, layer.ceiling + ticks_step, ticks_step)

    def draw(self, ax: Axes):
        """Draw colorbar for result layer

        Parameters
        ----------
        divider: mpl_toolkits.axes_grid1.axes_divider.AxesDivider
            AxesDivider which will make seperate axes for color bar. `Example for how to make divider <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>`_

        """
        cbar = plt.colorbar(
            self.mappable,
            cax=self._get_lax(ax),
            ticks=self.ticks,
            **self.settings.colorbar_kwargs,
        )
        cbar.set_label(self.settings.title, size=self.settings.colorbar_label_fontsize)
        cbar.ax.tick_params(labelsize=self.settings.colorbar_label_fontsize)


class RasterBaseMap(Layer):
    def __init__(
        self,
        shape: tuple,
        transform: rio.transform,
        crs: rio.crs.CRS,
        source=cx.providers.CartoDB.Voyager,
    ):
        """Basemap layer for use with raster layers

        Parameters
        ----------
        shape : tuple
            the shape of the raster that the basemap will be beneath
        transform
            the transform of the raster layer that the basemap will be beneath
        crs
            the crs for the basemap
        source
            contextily basemap provider
        """
        self.shape = shape
        self.transform = transform
        self.crs = crs
        self.source = source

    def draw(self, ax: plt.Axes):
        """Draw a basemap for plotting geospatial rasters on

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to draw basemap on
        """
        data = np.zeros(self.shape)
        data[0][0] = 1
        data[-1][-1] = 1

        # tmp is raster with just two dots which is used for contextily basemap
        ax.imshow(data, extent=plotting_extent(data, transform=self.transform))
        cx.add_basemap(ax=ax, crs=self.crs, source=self.source, interpolation="sinc")


class GeoMap:
    """Geographic map object for plotting geographic rasters.

    Examples
    --------
    >>> gmap = GeoMap()
    >>> result_layer = ContinuousRasterLayer(result_array,
                                         rio.transform.Affine(25.0, 0.0, 3412500.0, 0.0, -25.0, 5960000.0),
                                         "viridis")
    >>> vegetation_layer = CategoricalRasterLayer(management_array,
                                              rio.transform.Affine(2.5, 0.0, 3412500.0, 0.0, -2.5, 5960000.0),
                                              colorlist_mask)
    >>> vegetation_dict = {1:'N01.00',2:'N02.00'}
    >>> gmap.add_layer(BaseMap(result_array.shape, rio.transform.Affine(25.0, 0.0, 3412500.0, 0.0, -25.0, 5960000.0), CRS.from_epsg(28992))
    >>> gmap.add_layer(vegetation_layer)
    >>> gmap.add_layer(result_layer)
    >>> gmap.add_categorical_legend(vegetation_layer, vegetation_dict)
    >>> gmap.add_colorbar(result_layer, "this is very nice color bar")
    >>> gmap.title = "this is a very nice geographic map"
    >>> fig, ax = plt.subplots(1, 1, figsize=(12.75, 5))
    >>> gmap.draw(ax)
    """

    def __init__(self):
        #: list of features
        self.features = []

        #: draw a north arrow?
        self.north_arrow = False

        #: arrow location as (x, y, correction) and settings
        self.arrow_position = (0.96, 0.96, 0.075)
        self.arrow_kwargs = dict(
            arrowprops=dict(facecolor="black", width=4, headwidth=12.5),
            ha="center",
            va="center",
            fontsize=18,
            family="serif",
        )

        #: draw a draw_scalebar?
        self.draw_scalebar = False
        self.scalebar=ScaleBar(1, "m", location="upper left", box_alpha=0)

        #: the title for this GeoMap and settings
        self.title = "<Title>"
        self.title_kwargs = dict(weight="semibold", fontsize=15)

    def add_feature(self, feature: Layer or Legend):
        """Add a layer to the GeoMap

        Parameters
        ----------
        layer: Layer
            The layer to add to this GeoMAp

        """
        if isinstance(feature, RasterBaseMap):
            if len(self.features) == 0 or not isinstance(
                self.features[0], RasterBaseMap
            ):
                self.features.insert(0, feature)
            else:
                raise TypeError("Basemap is already defined")
        else:
            self.features.append(feature)

    def draw(self, ax: plt.Axes):
        """Draw basemap and raster layers in order and plot all _vector layers.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to plot everything on

        """
        for feature in self.features:
            feature.draw(ax)

        # plot and do all secondary shit like draw_scalebar, north arrow, turning off ticks etc on top of other
        # raster layers

        # draw draw_scalebar
        if self.draw_scalebar:
            ax.add_artist(self.scalebar)

        # draw north arrow
        if self.north_arrow:
            ax.annotate(
                "N",
                xy=(self.arrow_position[0], self.arrow_position[1]),
                xytext=(
                    self.arrow_position[0],
                    self.arrow_position[1] - self.arrow_position[2],
                ),
                xycoords=ax.transAxes,
                **self.arrow_kwargs,
            )

        # title and ticks
        ax.set_title(self.title, **self.title_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])


class Zoom:
    """Zoom object for easy-peasy automatic zooming to different parts of a geographic map

    Parameters
    ----------
    left: int
        left border of zoom
    right: int
        right border of zoom
    bottom: int
        bottom border
    top: int
        you catch my drift right?
    name: str
        name for this zoom, handy for automated exporting of maps
    """

    def __init__(self, left: int, right: int, bottom: int, top: int, name: str = ""):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.name = name
        self.dpi = 500

    def apply(self, ax):
        """

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to plot everything upon

        """
        ax.set_xlim(self.left, self.right)
        ax.set_ylim(self.bottom, self.top)
