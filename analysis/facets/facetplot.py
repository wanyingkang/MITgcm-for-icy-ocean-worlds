from logging import warning
import numpy as np
from matplotlib.pyplot import gca, sci, draw_if_interactive, figure
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.collections import Collection
from matplotlib.artist import allow_rasterization

__all__ = ['pcolormeshes', 'axpcolormeshes', 'MultiQuadMesh', 'backmask']

_NE = np.s_[..., 1:, 1:]
_NW = np.s_[..., 1:, :-1]
_SE = np.s_[..., :-1, 1:]
_SW = np.s_[..., :-1, :-1]

def backmask(X, Y):
    msk = []
    for x,y in zip(X, Y):
        # compute area as cross product of diagonals
        d1 = [x[_NE] - x[_SW], y[_NE] - y[_SW]]
        d2 = [x[_NW] - x[_SE], y[_NW] - y[_SE]]
#        med = np.median(ar.toglobal(map=0))
        finite = (np.isfinite(x[_NE]) &
                  np.isfinite(x[_NW]) &
                  np.isfinite(x[_SE]) &
                  np.isfinite(x[_SW]) &
                  np.isfinite(y[_NE]) &
                  np.isfinite(y[_NW]) &
                  np.isfinite(y[_SE]) &
                  np.isfinite(y[_SW]))
        idx = np.where(finite)
        ar = d1[0][idx]*d2[1][idx] - d1[1][idx]*d2[0][idx]
        m = ~finite
        # mask quads with area the opposite sign of the majority
#        np.put(m, idx, (ar/np.median(ar)) < -1.0)
        m[idx] = (ar/np.median(ar)) < -1.0
        msk.append(m)
    return msk


def pcolormeshes(X, Y, C, **kwargs):
    """
    Plot several quadrilateral meshes.

    Works like pcolormesh, but X, Y, C are sequences or Facets objects
    of possible arguments to pcolormesh.

    *C* may be a masked array, but *X* and *Y* may not.  Masked
    array support is implemented via *cmap* and *norm*; in
    contrast, :func:`~matplotlib.pyplot.pcolor` simply does not
    draw quadrilaterals with masked colors or vertices.

    *C* may be a Facets object or a sequence of arrays.  In the latter
    case, the arrays may represent RGB(A) values as NxMx3 or NxMx4 in
    which case cmap and norm are ignored.

    Keyword arguments:

      *cmap*: [ *None* | Colormap ]
        A :class:`matplotlib.colors.Colormap` instance. If *None*, use
        rc settings.

      *norm*: [ *None* | Normalize ]
        A :class:`matplotlib.colors.Normalize` instance is used to
        scale luminance data to 0,1. If *None*, defaults to
        :func:`normalize`.

      *vmin*/*vmax*: [ *None* | scalar ]
        *vmin* and *vmax* are used in conjunction with *norm* to
        normalize luminance data.  If either is *None*, it
        is autoscaled to the respective min or max
        of the color array *C*.  If not *None*, *vmin* or
        *vmax* passed in here override any pre-existing values
        supplied in the *norm* instance.

      *shading*: [ 'flat' | 'gouraud' ]
        'flat' indicates a solid color for each quad.  When
        'gouraud', each quad will be Gouraud shaded.  When gouraud
        shading, edgecolors is ignored.

      *edgecolors*: [*None* | ``'None'`` | ``'face'`` | color |
                     color sequence]
        If *None*, the rc setting is used by default.

        If ``'None'``, edges will not be visible.

        If ``'face'``, edges will have the same color as the faces.

        An mpl color or sequence of colors will set the edge color

      *alpha*: ``0 <= scalar <= 1``  or *None*
        the alpha blending value

      *maskback*: [ True | False ]
        mask quads with orientation opposite majority.

    Return value is a :class:`matplotlib.collections.MultiQuadMesh`
    object.

    kwargs can be used to control the
    :class:`matplotlib.collections.MultiQuadMesh` properties:

    %(MultiQuadMesh)s
    """
    if 'fig' in kwargs:
        figure(kwargs.pop('fig'))
    ax = gca()
#    washold = ax.ishold()
#    hold = kwargs.pop('hold', None)
#    if hold is not None:
#        ax.hold(hold)
    clear = kwargs.pop('cla', False)
    if clear:
        ax.cla()
    if hasattr(ax, 'projection') and not 'transform' in kwargs:
        from cartopy import crs as ccrs
        kwargs['transform'] = ccrs.PlateCarree()
#    try:
    ret = axpcolormeshes(ax, X, Y, C, **kwargs)
    draw_if_interactive()
#    finally:
#        ax.hold(washold)
    sci(ret)
    return ret


def axpcolormeshes(ax, X, Y, C, **kwargs):
#    if not ax._hold:
#        ax.cla()

    alpha = kwargs.pop('alpha', None)
    norm = kwargs.pop('norm', None)
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    shading = kwargs.pop('shading', 'flat').lower()
    antialiased = kwargs.pop('antialiased', False)
    maskback = kwargs.pop('maskback', True)
    kwargs.setdefault('edgecolors', 'None')

    assert shading != 'gouraud'

    try:
        Xl = X.facets
    except AttributeError:
        Xl = X

    try:
        Yl = Y.facets
    except AttributeError:
        Yl = Y

    Ny = [x.shape[0]-1 for x in Xl]
    Nx = [x.shape[1]-1 for x in Xl]
    sizes = [(x.shape[0]-1)*(x.shape[1]-1) for x in Xl]
    inds = np.cumsum([0] + sizes)
    Xsizes = [(x.shape[0])*(x.shape[1]) for x in Xl]
    Xinds = np.cumsum([0] + Xsizes)
    n = len(Xl)

    try:
        Cl = C.facets
    except AttributeError:
        Cl = C

    if maskback:
        msk = backmask(Xl, Yl)
        if Cl[0].ndim > 2:
            for f in range(n):
                Cl[f][3, msk[f]] = 0.
        else:
            try:
                C.mask |= msk
            except (AttributeError, ValueError):
                try:
                    for i in range(n):
                        Cl[i].mask |= msk[i]
                except AttributeError:
                    C = Cl = [np.ma.MaskedArray(Cl[i], msk[i]) for i in range(n)]
#        print Cl[0].shape

    # convert to one dimensional array
    try:
        C = C.toglobal(map=0)
    except AttributeError:
        warning('Not using toglobal, may not be a FacetArray')
        tmp = np.zeros((sum(sizes),)+Cl[0].shape[2:], Cl[0].dtype).view(Cl[0].__class__)
        for i in range(len(sizes)):
            tmp[inds[i]:inds[i+1]] = Cl[i].reshape((-1,) + Cl[i].shape[2:])
        C = tmp
    else:
        if C.ndim > 1:
            C = C.T

    coordsbase = np.zeros((sum(Xsizes), 2), dtype=float)
    coords = [coordsbase[Xinds[i]:Xinds[i+1]].reshape(Xl[i].shape+(2,)) for i in range(n)]
    for i in range(n):
        coords[i][:,:,0] = Xl[i]
        coords[i][:,:,1] = Yl[i]

    collection = MultiQuadMesh(
        Nx, Ny, coords,
        antialiased=antialiased, shading=shading, **kwargs)
    collection.set_alpha(alpha)
    if C.ndim > 1:
        collection.set_facecolors(C)
    else:
        collection.set_array(C)
        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
        collection.autoscale_None()

    ax.grid(False)

    # Transform from native to data coordinates?
    t = collection._transform
    if (not isinstance(t, mtransforms.Transform)
        and hasattr(t, '_as_mpl_transform')):
        t = t._as_mpl_transform(ax.axes)

    for i in range(len(sizes)):
        if t and any(t.contains_branch_seperately(ax.transData)):
            trans_to_data = t - ax.transData
            pts = np.vstack([Xl[i], Yl[i]]).T.astype(np.float)
            transformed_pts = trans_to_data.transform(pts)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
        else:
            x = Xl[i]
            y = Yl[i]

        if i == 0:
            minx = np.amin(x)
            maxx = np.amax(x)
            miny = np.amin(y)
            maxy = np.amax(y)
        else:
            minx = min(minx, np.amin(x))
            maxx = max(maxx, np.amax(x))
            miny = min(miny, np.amin(y))
            maxy = max(maxy, np.amax(y))

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection


class MultiQuadMesh(Collection):
    """
    Class for the efficient drawing of multiple quadrilateral meshes.
    
    Same as QuadMesh, but draw several meshes, controlled by the same
    ScalarMappable.

    Only flat shading is supported right now.
    """
    def __init__(self, meshWidth, meshHeight, coordinates,
                 antialiased=True, shading='flat', **kwargs):
        Collection.__init__(self, **kwargs)
        self._n = len(meshWidth)
        self._meshWidth = meshWidth
        self._meshHeight = meshHeight
        self._antialiased = antialiased
        self._shading = shading
        # this will be used for indexing the flat scalar array
        sizes = [h*w for h,w in zip(meshHeight, meshWidth)]
        self._i = np.cumsum([0] + sizes)

        # By converting to floats now, we can avoid that on every draw.
        self._coordinates = [np.asfarray(c).reshape(h+1, w+1, 2) 
            for c,h,w in zip(coordinates,meshHeight,meshWidth)]

        self._bbox = mtransforms.Bbox.unit()
        for c in coordinates:
            self._bbox.update_from_data_xy(c.reshape(-1, 2))
            self._bbox.ignore(False)
        self._bbox.ignore(True)

    def get_datalim(self, transData):
        return self._bbox

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, self.get_gid())
        transform = self.get_transform()
        transOffset = self.get_offset_transform()
        offsets = self._offsets

        if self.have_units():
            if len(self._offsets):
                xs = self.convert_xunits(self._offsets[:, 0])
                ys = self.convert_yunits(self._offsets[:, 1])
                offsets = zip(xs, ys)

        offsets = np.asarray(offsets, np.float_)
        offsets.shape = (-1, 2)                 # Make it Nx2

        self.update_scalarmappable()

        if not transform.is_affine:
            coordinates = [transform.transform(
                coords.reshape((coords.shape[0]*coords.shape[1], 2))
                ).reshape(coords.shape) for coords in self._coordinates]
            transform = mtransforms.IdentityTransform()
        else:
            coordinates = self._coordinates

        if not transOffset.is_affine:
            offsets = transOffset.transform_non_affine(offsets)
            transOffset = transOffset.get_affine()

        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])

        if self._shading == 'gouraud':
            triangles, colors = self.convert_mesh_to_triangles(
                self._meshWidth, self._meshHeight, coordinates)
            renderer.draw_gouraud_triangles(
                gc, triangles, colors, transform.frozen())
        else:
            for i in range(self._n):
                renderer.draw_quad_mesh(
                    gc, transform.frozen(), self._meshWidth[i], self._meshHeight[i],
                    coordinates[i], offsets, transOffset, self._get_facecolor(i),
                    self._antialiased, self._get_edgecolor(i))
        gc.restore()
        renderer.close_group(self.__class__.__name__)

    def _get_facecolor(self, i):
        return self._facecolors[self._i[i]:self._i[i+1]]

    def _get_edgecolor(self, i):
        if self._edgecolors == 'face':
            return self.get_facecolors(i)
        else:
            return self._edgecolors[self._i[i]:self._i[i+1]]


