import rasterio as rio
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from rasterio.plot import plotting_extent
import contextily as cx
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import geopandas as gp
import matplotlib.patches as mpatches
import numpy as np
from abc import abstractmethod, ABC


class Layer(ABC):
    """Layer parent class"""

    def __init__(self):
        #: is this layer visible?
        self.visible = True

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
        super().__init__()
        self.raster = raster
        self.transform = transform
        self.cmap = "viridis"

        #: ceiling of raster values
        self.ceiling = np.ceil(np.nanmax(self.raster))
        if np.isnan(self.ceiling):
            self.ceiling = 1

    def draw(self, ax: plt.Axes):
        """Draw management type map as background

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to draw layer on
        """
        cmap = cm.get_cmap(self.cmap)
        cmap.set_bad(alpha=0)
        if self.visible:

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
    cmap_stepsize : int
        the number of steps between each whole number. default is 10, so that the colormap is divided into steps of 0.1
    """

    def __init__(
        self,
        raster: np.array,
        transform: rio.transform,
        cmap: str or list = "viridis",
        cmap_stepsize: int = 10,
    ):
        super().__init__(raster, transform)

        #: make normalization
        self.norm = colors.Normalize(vmin=0, vmax=self.ceiling)

        # specify total number of steps
        number_of_steps = int(self.ceiling * abs(cmap_stepsize))

        #: colormap for continuous raster layer
        self.cmap = cm.get_cmap(cmap, number_of_steps)


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
        if self.ceiling <= 1:
            self.norm = plt.Normalize(0, 1)
        else:
            self.norm = colors.BoundaryNorm(
                boundaries=np.arange(1, self.ceiling + 1) - 0.5, ncolors=self.ceiling
            )
        #: colormap for this layer
        self.cmap = colors.ListedColormap(colorlist)


class VectorLayer(Layer):
    """Vector layer

    Parameters
    ----------
    vector: geopandas.GeoDataFrame
        GeodataFrame containing the vector
    kwargs
        Keyword arguments to set layer properties
    """

    def __init__(self, vector: gp.GeoDataFrame, **kwargs):
        super().__init__()
        self.vector = vector
        self.facecolor = "none"
        self.kwargs = kwargs
        if "norm" in kwargs:
            self.norm = kwargs["norm"]
        if "cmap" in kwargs:
            self.cmap = kwargs["cmap"]

    def draw(self, ax):
        """Draw this vector layer on the Axes

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to plot everything upon

        """
        self.vector.plot(ax=ax, aspect=1, **self.kwargs)


class MtLegend:
    """Management type legend class

    Parameters
    ----------
    categorical_raster : CategoricalRasterLayer
        Management type layer with management types as integers
    dictionary : dict
        Dictionary to translate integer values to labels or landscape type codes
    """

    def __init__(self, categorical_raster: CategoricalRasterLayer, dictionary: dict):
        self.norm = categorical_raster.norm
        self.dictionary = dictionary
        self.cmap = categorical_raster.cmap
        self.visible = categorical_raster.visible

    def draw(self, divider: AxesDivider):
        """

        Parameters
        ----------
        divider
            AxesDivider which will make seperate axes for management type legend. `Example for how to make divider <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>`_


        """
        if self.visible:
            # management_map legend in seperate axes
            lax = divider.append_axes("left", size=0.1, pad=0.08)
            lax.axis("off")
            mtypes = list(self.dictionary.keys())

            # create a patch (proxy artist) for every color
            patches = [
                mpatches.Patch(
                    color=self.cmap(self.norm(mtype)),
                    label=self.dictionary[mtype],
                )
                for mtype in mtypes
            ]
            # put those patched as legend-handles into the legend
            legend = lax.legend(
                handles=patches,
                bbox_to_anchor=(-9.5, 0.5),
                loc=10,
                borderaxespad=0.0,
                ncol=2,
                edgecolor=[0, 0, 0, 1],
                prop={"size": 8},
            )
            legend.get_frame().set_linewidth(0.8)


class ColorBar:
    """A color bar, pretty straightforward

    Parameters
    ----------
    continuous_layer
        Raster layer to take normalization and colormap from
    label: str
        Label vor the color bar
    """

    def __init__(self, continuous_layer:ContinuousRasterLayer, label: str):
        self.mappable = cm.ScalarMappable(continuous_layer.norm, continuous_layer.cmap)
        self.label = label
        self.visible = continuous_layer.visible
        self.ticks = np.arange(0,continuous_layer.ceiling+1,1)

    def draw(self, divider: AxesDivider):
        """Draw colorbar for result layer

        Parameters
        ----------
        divider: mpl_toolkits.axes_grid1.axes_divider.AxesDivider
            AxesDivider which will make seperate axes for color bar. `Example for how to make divider <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>`_

        """
        cax = divider.append_axes("right", size="2%", pad=0.08)
        cbar = plt.colorbar(self.mappable, cax=cax, aspect=25, ticks=self.ticks)
        cbar.set_label(self.label, size=20)
        cbar.ax.tick_params(labelsize=17.5)


class BaseMap(Layer):
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
        super().__init__()
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
        if self.visible:
            data = np.zeros(self.shape)
            data[0][0] = 1
            data[-1][-1] = 1

            # tmp is raster with just two dots which is used for contextily basemap
            ax.imshow(data, extent=plotting_extent(data, transform=self.transform))
            cx.add_basemap(ax=ax, crs=self.crs, source=self.source, zoom=11)


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
        #: list of layers
        self._layers = []

        #: list of legends
        self._legends = []

        #: plot basemap labels over all other layers?
        self.do_basemap_labels = False

        #: basmap label provider
        self.basemap_labels = cx.providers.Stamen.TonerLabels

        #: draw a north arrow?
        self.north_arrow = False

        #: arrow location as (x, y, correction)
        self.arrow_parameters = (0.96, 0.96, 0.075)

        #: draw a scalebar?
        self.scalebar = False

        #: the title for this GeoMap
        self.title = "<Title>"

    def add_layer(self, layer: Layer):
        """Add a layer to the GeoMap

        Parameters
        ----------
        layer: Layer
            The layer to add to this GeoMAp

        """
        if isinstance(layer, BaseMap):
            if len(self._layers) == 0 or not isinstance(self._layers[0], BaseMap):
                self._layers.insert(0, layer)
            else:
                raise TypeError("Basemap is already defined")
        else:
            self._layers.append(layer)

    def add_categorical_legend(
        self, categorical_layer: CategoricalRasterLayer, dictionary: dict
    ):
        """Add categorical Legend to GeoMap

        Parameters
        ----------
        categorical_layer
        dictionary

        """
        self._legends.append(MtLegend(categorical_layer, dictionary))

    def add_colorbar(self, continuous_layer, label: str):
        """Add a color bar to this GeoMap

        Parameters
        ----------
        continuous_layer
            continuous layer to get colormap and normalization from
        label: str
            what does this bar say?
        """
        self._legends.append(ColorBar(continuous_layer, label))

    def draw(self, ax: plt.Axes):
        """Draw basemap and raster layers in order and plot all vector layers.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            Axes object to plot everything on

        """
        for layer in self._layers:
            if layer.visible:
                layer.draw(ax)

        divider = make_axes_locatable(ax)
        for legend in self._legends:
            if legend.visible:
                legend.draw(divider)

        # plot and do all secondary shit like labels, scalebar, north arrow, turning off ticks etc on top of other
        # raster layers
        # labels
        if self.do_basemap_labels:
            cx.add_basemap(ax=ax, crs=self._layers[0].crs, source=self.basemap_labels)

        # draw scalebar
        if self.scalebar:
            ax.add_artist(ScaleBar(1, "m", location="upper left", box_alpha=0))

        # draw north arrow
        if self.north_arrow:
            ax.annotate(
                "N",
                xy=(self.arrow_parameters[0], self.arrow_parameters[1]),
                xytext=(
                    self.arrow_parameters[0],
                    self.arrow_parameters[1] - self.arrow_parameters[2],
                ),
                arrowprops=dict(facecolor="black", width=4, headwidth=12.5),
                ha="center",
                va="center",
                fontsize=18,
                family="serif",
                xycoords=ax.transAxes,
            )

        # title and ticks
        ax.set_title(self.title, weight="semibold", fontsize=15)
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

    def __init__(self, left: int, right: int, bottom: int, top: int, name: str):
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
