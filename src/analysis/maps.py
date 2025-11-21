## Code provided by S. Heydenreich, adapted to the purposes of this analysis.
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.collections import PolyCollection


def create_sky_coord(ra, dec, deg=True, ra_u=None, dec_u=None):
    """
    Creates SkyCoord object for match_catalogs function.

    Parameters
    ----------
    ra : (list) or (str)
        List containing RAs of the objects.
        Or RA of the object.
    dec : (list) or (str)
        List containing Decs of the objects.
        Or Dec of the object.
    deg : (bool)
        Is the coordinate already in degrees?
    ra_u : (astropy.unit) or (None)
        RA unit.
    dec_u : (astropy.unit) or (None)
        Dec unit.


    Returns
    -------
    sky_coord : (astropy.coordinates.SkyCoord)
        SkyCoord object containing positional information,
        including units and separation (dimensionless, set to 1).
    """
    if deg == True:
        sky_coord = SkyCoord(ra * u.deg, dec * u.deg)
    else:
        sky_coord = SkyCoord(ra, dec, unit=(ra_u, dec_u))

    return sky_coord


def put_survey_on_grid(
    ra, dec, ra_proj, dec_proj, pixels, vertices, unit="deg", smoothing=0.4 * u.deg
):
    c_survey = SkyCoord(ra, dec, unit=unit)
    c_footprint = SkyCoord(ra_proj, dec_proj, unit=unit)
    idx, sep2d, dist3d = c_footprint.match_to_catalog_sky(c_survey)
    inside = sep2d < smoothing
    return pixels[inside], vertices[inside], inside


def get_vertices_from_pixels(pixels, inside, nside):
    vertices = np.zeros((len(pixels), 4, 2))
    vertices[:, :, 0] = hp.pix2ang(nside, pixels, nest=False, lonlat=True)[0]
    vertices[:, :, 1] = hp.pix2ang(nside, pixels, nest=False, lonlat=True)[1]
    return vertices[inside]


def vertex_with_edge(skmcls, vertices, color=None, vmin=None, vmax=None, **kwargs):
    """Plot polygons (e.g. Healpix vertices)

    Args:
        vertices: cell boundaries in RA/Dec, from getCountAtLocations()
        color: string or matplib color, or numeric array to set polygon colors
        vmin: if color is numeric array, use vmin to set color of minimum
        vmax: if color is numeric array, use vmin to set color of minimum
        **kwargs: matplotlib.collections.PolyCollection keywords
    Returns:
        matplotlib.collections.PolyCollection
    """
    vertices_ = np.empty_like(vertices)
    vertices_[:, :, 0], vertices_[:, :, 1] = skmcls.proj.transform(
        vertices[:, :, 0], vertices[:, :, 1]
    )

    # remove vertices which are split at the outer meridians
    # find variance of vertice nodes large compared to dispersion of centers
    centers = np.mean(vertices, axis=1)
    x, y = skmcls.proj.transform(centers[:, 0], centers[:, 1])
    var = np.sum(np.var(vertices_, axis=1), axis=-1) / (x.var() + y.var())
    sel = var < 0.05
    vertices_ = vertices_[sel]

    from matplotlib.collections import PolyCollection

    zorder = kwargs.pop("zorder", 0)  # same as for imshow: underneath everything
    rasterized = kwargs.pop("rasterized", True)
    alpha = kwargs.pop("alpha", 1)
    # if alpha < 1:
    #     lw = kwargs.pop('lw', 0)
    # else:
    #     lw = kwargs.pop('lw', None)
    coll = PolyCollection(
        vertices_, zorder=zorder, rasterized=rasterized, alpha=alpha, **kwargs
    )
    if color is not None:
        coll.set_array(color[sel])
        coll.set_clim(vmin=vmin, vmax=vmax)
    # coll.set_edgecolor("face")
    skmcls.ax.add_collection(coll)
    skmcls.ax.set_rasterization_zorder(zorder)
    return coll


def get_boundary_mask(vertices, nside, niter=1):
    boundary_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[vertices] = 1
    assert niter > 0
    neighbors = vertices
    for i in range(niter):
        neighbors = hp.get_all_neighbours(nside, neighbors)
        neighbors = np.unique(neighbors, axis=1)
    boundary_mask[neighbors] = 1

    boundary_mask = boundary_mask & (~mask)

    return boundary_mask


def get_fsky(input_mask, threshold=0.1):
    """get the fraction of the observable sky

    Parameters
    ---------
    input_mask: np.ndarray
        healpy array indicating the input mask (0: masked, 1: visible)
    threshold: int
        mask cutoff value
    """
    if np.issubdtype(input_mask.dtype, np.bool_):
        return float(np.sum(input_mask)) / len(input_mask)
    return len(input_mask[input_mask > threshold]) / len(input_mask)


def estimate_sky_coverage(ras, decs, nside=1024):
    phi, theta = np.radians(ras), np.radians(90.0 - decs)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    mask = np.zeros(hp.nside2npix(nside))
    mask[ipix] = 1
    return get_fsky(mask)


def make_cover_map(ras, decs, nside=1024):
    phi, theta = np.radians(ras), np.radians(90.0 - decs)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[ipix] = 1
    return mask


def get_area(input_mask):
    return get_fsky(input_mask) * 4 * np.pi * (180 / np.pi) ** 2


def get_overlap(mask1, mask2):
    return np.logical_and(mask1, mask2)
