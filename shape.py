#!/usr/bin/python
# -*- coding: utf-8 -*-

# shapely_hatch - simple hatching function demo
# produces WKT of a 45° hatched crescent to stdout
# scruss — 2014-04-13 — WTFPL (srsly)

from shapely.geometry import box, MultiLineString, Point
from shapely.affinity import rotate
from shapely import speedups
from math import sqrt

# enable Shapely speedups, if possible
if speedups.available:
    speedups.enable()

def hatchbox(rect, angle, spacing):
    """
    returns a Shapely geometry (MULTILINESTRING, or more rarely,
    GEOMETRYCOLLECTION) for a simple hatched rectangle.

    args:
    rect - a Shapely geometry for the outer boundary of the hatch
           Likely most useful if it really is a rectangle

    angle - angle of hatch lines, conventional anticlockwise -ve

    spacing - spacing between hatch lines

    GEOMETRYCOLLECTION case occurs when a hatch line intersects with
    the corner of the clipping rectangle, which produces a point
    along with the usual lines.
    """

    (llx, lly, urx, ury) = rect.bounds
    centre_x = (urx + llx) / 2
    centre_y = (ury + lly) / 2
    diagonal_length = sqrt((urx - llx) ** 2 + (ury - lly) ** 2)
    number_of_lines = 2 + int(diagonal_length / spacing)
    hatch_length = spacing * (number_of_lines - 1)

    # build a square (of side hatch_length) horizontal lines
    # centred on centroid of the bounding box, 'spacing' units apart
    coords = []
    for i in range(number_of_lines):
        # alternate lines l2r and r2l to keep HP-7470A plotter happy ☺
        if i % 2:
            coords.extend([((centre_x - hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          + hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
        else:
            coords.extend([((centre_x + hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          - hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
    # turn array into Shapely object
    lines = MultiLineString(coords)
    # Rotate by angle around box centre
    lines = rotate(lines, angle, origin='centroid', use_radians=False)
    # return clipped array
    return rect.intersection(lines)

# pipe-separated output; can be read by QGIS
print('ID| WKT')
page = box(1000, 1000, 6000, 6000)
hatching = hatchbox(page, 45, 50)
circle = Point(2500, 2500).buffer(1000)
circle1 = Point(2000, 2500).buffer(500)
crescent = circle.difference(circle1)
crescent_hatch = crescent.intersection(hatching)
print('1|', crescent_hatch)