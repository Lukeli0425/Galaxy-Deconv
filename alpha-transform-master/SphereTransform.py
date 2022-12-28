r"""
This module contains two functions which can be used to convert functions
defined on the sphere (given in the form of a HEALPix map) to an array of
12 cartesian images, where each of the cartesian images corresponds to a
pullback of the given function with respect to one of the 12 HEALPix charts.

Since the HEALPix charts are area preserving, taking the sphere-based
alpha-shearlet transform is the same as taking the cartesian alpha-shearlet
transform of each of the 12 cartesian images and concatenating the results.

Reconstruction from these (possibly modified, e.g. thresholded) coefficients
is straightforward: Reconstruct each of the 12 cartesian images, concatenate
the results and push the result back to the sphere, using
:func:`put_all_faces`, which again results in a HEALPix map.
"""

import healpy as hp
import numpy as np
import math


def get_all_faces(Imag, nested=False):
    r"""
    This function maps a function defined on the sphere to an array of 12
    cartesian images, where each of the cartesian images corresponds to
    a pullback of the given function with respect to one of the 12 HEALPix
    charts.

    **Required parameter**

    :param numpy.ndarray Imag:
       The function defined on the sphere, in the form of a one-dimensional
       :class:`numpy.ndarray` (a HEALPix map).

       .. warning::
           If the HEALPix map ``Imag`` uses "NESTED" ordering, the parameter
           ``nested`` needs to be set to ``True``.

    **Keyword parameter**

    :param bool nested:
        This parameter determines the ordering with which the different
        HEALPix pixels are stored in the array ``Imag``; see
        http://healpix.jpl.nasa.gov/html/intronode4.htm for more details.

        If ``nested`` is set to ``False``, this signifies that ``Imag`` uses
        the "RING" ordering.

    **Return value**

    :return:
        A 3-dimensional :class:`numpy.ndarray` consisting of 12 cartesian
        images (2-dimensional arrays), where each image is a pullback of
        the input function ``Imag`` with respect to one of  the 12 HEALPix
        charts.
    """
    npix = np.shape(Imag)[0]
    assert npix % 12 == 0
    nside = hp.npix2nside(npix)
    taille_face = npix // 12
    cote = int(math.sqrt(taille_face))
    CubeFace = np.zeros((12, cote, cote))
    if not nested:
        NewIm = hp.reorder(Imag, r2n=True)
    else:
        NewIm = Imag
    index = np.zeros((cote, cote))
    index = np.array([hp.xyf2pix(nside, x, y, 0, True)
                      for x in range(nside)
                      for y in range(nside)])
    for face in range(12):
        # print("Process Face {0}".format(face))
        CubeFace[face] = np.resize(NewIm[index + taille_face * face],
                                   (cote, cote))
        # plt.figure(),imshow(np.log10(1+CubeFace[face,:,:]*1e6))
        # plt.title("face {0}".format(face)),plt.colorbar()
    return CubeFace


def put_all_faces(CubeFace, nested=False):
    r"""
    Given 12 cartesian functions corresponding to the 12 HEALPix
    faces,  this function pushes each of them back to the sphere
    using one of the HEALPix charts. The 12 different functions are
    pasted together to obtain a function defined on all of the sphere.

    **Required parameter**

    :param numpy.ndarray CubeFace:
        A 3-dimensional :class:`numpy.ndarray` consisting of 12 cartesian
        images (2-dimensional arrays) corresponding to the 12 HEALPix faces.

    **Keyword parameter**

    :param bool nested:
        This parameter determines the ordering with which the different
        HEALPix pixels are stored in the returned array. See
        http://healpix.jpl.nasa.gov/html/intronode4.htm for more details.

        If ``nested`` is set to ``False``, this signifies that the return value
        should use the "RING" ordering.

    **Return value**

    :return:
        A one-dimensional :class:`numpy.ndarray` (a HEALPix map)
        corresponding to the resulting "pasted" function.
    """
    npix = np.size(CubeFace)
    assert npix % 12 == 0
    nside = hp.npix2nside(npix)
    taille_face = npix // 12
    cote = int(math.sqrt(taille_face))
    Imag = np.zeros((npix))
    index = np.zeros((cote, cote))
    index = np.array([hp.xyf2pix(nside, x, y, 0, True)
                      for x in range(nside)
                      for y in range(nside)])
    for face in range(12):
        # print("Process Face {0}".format(face))
        Imag[index + taille_face * face] = np.resize(CubeFace[face],
                                                     (cote, cote))

    if not nested:
        NewIm = hp.reorder(Imag, n2r=True)
    else:
        NewIm = Imag
    return NewIm
