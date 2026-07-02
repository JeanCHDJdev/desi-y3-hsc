"""
This file contains methods to interact with healpy and plot maps
with healpy and matplotlib.
"""

import numpy as np
import healpy as hp
import astropy.units as u

from astropy.coordinates import SkyCoord
from matplotlib.collections import PolyCollection


def put_survey_on_grid(ra, dec, ra_proj, dec_proj, pixels, vertices, unit="deg", smoothing=0.4 * u.deg):
    """
    Match survey coordinates to healpy pixels and vertices.

    Parameters
    ----------
    ra: np.ndarray
        Array of right ascension values of the survey in degrees.
    dec: np.ndarray
        Array of declination values of the survey in degrees.
    ra_proj: np.ndarray
        Array of right ascension values of the projected footprint in degrees.
    dec_proj: np.ndarray
        Array of declination values of the projected footprint in degrees.
    pixels: np.ndarray
        Array of healpy pixel indices.
    vertices: np.ndarray
        Array of shape (N, 4, 2) representing the vertices of the pixels.
    unit: str
        Unit of the input coordinates (default: 'deg').
    smoothing: astropy.units.Quantity
        Smoothing angle to determine if a pixel is inside the survey area. (Default: 0.4 deg)

    Returns
    -------
    np.ndarray
        Array of healpy pixel indices that are inside the survey area.
    np.ndarray
        Array of shape (M, 4, 2) representing the vertices of the pixels
        that are inside the survey area.
    np.ndarray
        Boolean array indicating which pixels are inside the survey area.
    """

    c_survey = SkyCoord(ra, dec, unit=unit)
    c_footprint = SkyCoord(ra_proj, dec_proj, unit=unit)
    idx, sep2d, dist3d = c_footprint.match_to_catalog_sky(c_survey)
    inside = sep2d < smoothing
    return pixels[inside], vertices[inside], inside


def vertices_from_pixels(pixels, inside, nside):
    """
    Get the vertices of the given healpy pixels.

    Parameters
    ----------
    pixels: np.ndarray
        Array of healpy pixel indices.
    inside: np.ndarray
        Boolean array indicating which pixels are inside the survey area.
    nside: int
        Healpy nside parameter.

    Returns
    -------
    np.ndarray
        Array of shape (N, 4, 2) representing the vertices of the pixels
        that are inside the survey area.
    """
    vertices = np.zeros((len(pixels), 4, 2))
    vertices[:, :, 0] = hp.pix2ang(nside, pixels, nest=False, lonlat=True)[0]
    vertices[:, :, 1] = hp.pix2ang(nside, pixels, nest=False, lonlat=True)[1]
    return vertices[inside]


def vertex_with_edge(skmcls, vertices, var_threshold=0.05, color=None, vmin=None, vmax=None, **kwargs):
    """
    Plot polygons (e.g. Healpix vertices)

    Parameters
    ----------
    skmcls: SkyMap
        SkyMap class instance.
    vertices: np.ndarray
        Array of shape (N, M, 2) representing N polygons with M vertices each.
    var_threshold: float
        Variance threshold to filter out polygons split at the outer meridians. Default is 0.05.
    color: np.ndarray
        Array of shape (N,) representing the color for each polygon. If None, no color is applied.
    vmin: float
        Minimum value for color scaling. If None, the minimum of the color array is used.
    vmax: float
        Maximum value for color scaling. If None, the maximum of the color array is used.
    **kwargs: dict
        Additional keyword arguments for PolyCollection.

    Returns
    -------
    PolyCollection
        The created PolyCollection object.
    """
    vertices_ = np.empty_like(vertices)
    vertices_[:, :, 0], vertices_[:, :, 1] = skmcls.proj.transform(vertices[:, :, 0], vertices[:, :, 1])

    # remove vertices which are split at the outer meridians
    # find variance of vertice nodes large compared to dispersion of centers
    centers = np.mean(vertices, axis=1)
    x, y = skmcls.proj.transform(centers[:, 0], centers[:, 1])
    var = np.sum(np.var(vertices_, axis=1), axis=-1) / (x.var() + y.var())
    sel = var < var_threshold
    vertices_ = vertices_[sel]

    zorder = kwargs.pop("zorder", 0)
    rasterized = kwargs.pop("rasterized", True)
    alpha = kwargs.pop("alpha", 1)
    coll = PolyCollection(vertices_, zorder=zorder, rasterized=rasterized, alpha=alpha, **kwargs)
    if color is not None:
        coll.set_array(color[sel])
        coll.set_clim(vmin=vmin, vmax=vmax)

    skmcls.ax.add_collection(coll)
    skmcls.ax.set_rasterization_zorder(zorder)
    return coll


def sky_boundary_mask(vertices, nside, niter=1):
    """
    Get a boundary mask given the input vertices.

    Parameters
    ----------
    vertices: np.ndarray
        Array of healpy pixel indices representing the vertices.
    nside: int
        Healpy nside parameter.
    niter: int
        Number of iterations to expand the boundary.

    Returns
    -------
    np.ndarray
        Boolean array representing the boundary mask.
    """
    boundary_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[vertices] = 1

    assert niter > 0
    neighbors = vertices
    for _ in range(niter):
        neighbors = hp.get_all_neighbours(nside, neighbors)
        neighbors = np.unique(neighbors, axis=1)

    boundary_mask[neighbors] = 1
    boundary_mask = boundary_mask & (~mask)
    return boundary_mask


def get_fsky(input_mask, threshold=0.1):
    """
    Get the fraction of the observable sky.

    Parameters
    ---------
    input_mask: np.ndarray
        healpy array indicating the input mask (0: masked, 1: visible)
    threshold: int
        mask cutoff value

    Returns
    -------
    float
        Fraction of the observable sky.
    """
    if np.issubdtype(input_mask.dtype, np.bool_):
        return float(np.sum(input_mask)) / len(input_mask)
    return len(input_mask[input_mask > threshold]) / len(input_mask)


def estimate_sky_coverage(ras, decs, nside=1024):
    """
    Estimate the sky coverage from the given RA/Dec coordinates.

    Parameters
    ----------
    ras: np.ndarray
        Array of right ascension values in degrees.
    decs: np.ndarray
        Array of declination values in degrees.
    nside: int
        Healpy nside parameter.

    Returns
    -------
    float
        Estimated fraction of the observable sky.
    """
    phi, theta = np.radians(ras), np.radians(90.0 - decs)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    mask = np.zeros(hp.nside2npix(nside))
    mask[ipix] = 1
    return get_fsky(mask)


def sky_cover_map(ras, decs, nside=1024):
    """
    Create a coverage map from the given RA/Dec coordinates.

    Parameters
    ----------
    ras: np.ndarray
        Array of right ascension values in degrees.
    decs: np.ndarray
        Array of declination values in degrees.
    nside: int
        Healpy nside parameter.

    Returns
    -------
    np.ndarray
        Healpy boolean array representing the coverage map.
    """
    phi, theta = np.radians(ras), np.radians(90.0 - decs)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[ipix] = 1
    return mask


def sky_area(input_mask):
    """
    Get the area of the sky covered by the input mask in square degrees.

    Parameters
    ---------
    input_mask: np.ndarray
        healpy array indicating the input mask (0: masked, 1: visible)

    Returns
    -------
    float
        Area of the sky covered by the input mask in square degrees.
    """
    return get_fsky(input_mask) * 4 * np.pi * (180 / np.pi) ** 2


def sky_overlap(mask1, mask2):
    """
    Get the overlapping area between two healpy masks.

    Parameters
    ---------
    mask1: np.ndarray
        healpy array indicating the first mask (0: masked, 1: visible)
    mask2: np.ndarray
        healpy array indicating the second mask (0: masked, 1: visible)

    Returns
    -------
    np.ndarray
        Boolean array representing the overlapping area between the two masks.
    """
    return np.logical_and(mask1, mask2)