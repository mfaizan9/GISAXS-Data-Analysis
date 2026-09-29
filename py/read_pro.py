"""Instrument parameters read straight out of a BornAgain 1.7 .pro (XML) project file.

Nothing about the instrument is transcribed by hand, so a notebook that calls
``read_pro`` cannot drift away from the project file that ships beside it.

    from read_pro import read_pro
    PRO = read_pro("Bornagain/various-angles_000.pro")
    PRO["alpha_i"], PRO["nx"], PRO["ny"]
"""

import xml.etree.ElementTree as ET

import numpy as np


def read_pro(path):
    """Return the instrument parameters of a BornAgain .pro file as a dict.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .pro (XML) project file.

    Returns
    -------
    dict with keys
        nx, ny       : detector bins along x and y                  [int]
        width, height: declared detector extent = (Nbins-1)*pitch   [mm]
        u0, v0       : foot of the detector normal in (u, v)        [mm]
        N            : detector normal vector, sample -> detector   [mm]
        wavelength   : beam wavelength                              [nm]
        alpha_i      : beam inclination (incident) angle            [deg]
    """
    root = ET.parse(path).getroot()

    def item(node, **attrs):
        for it in node.iter("Item"):
            if all(it.get(k) == v for k, v in attrs.items()):
                return it
        raise KeyError(f"{path}: no <Item> with {attrs}")

    def value(node, tag):
        return float(item(node, Tag=tag).find("Parameter").get("ParValue"))

    det = item(root, ModelType="RectangularDetector")
    xax = item(det, Tag="X axis")
    yax = item(det, Tag="Y axis")
    nrm = item(det, Tag="Normal vector")

    return dict(
        nx=int(value(xax, "Nbins")),
        ny=int(value(yax, "Nbins")),
        width=value(xax, "Max"),           # = (Nbins-1)*pitch, NOT Nbins*pitch
        height=value(yax, "Max"),
        N=np.array([value(nrm, "X"), value(nrm, "Y"), value(nrm, "Z")]),
        u0=value(det, "u0"),
        v0=value(det, "v0"),
        wavelength=value(item(root, ModelType="BeamWavelength"), "Value"),
        alpha_i=value(item(root, ModelType="BeamInclinationAngle"), "Value"),
    )
