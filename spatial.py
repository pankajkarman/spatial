#!/usr/bin/env python
# coding: utf-8

import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
from scipy.interpolate import griddata

class IDW:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def interpolate(self, xi, yi):
        xi, yi = np.meshgrid(xi, yi)
        ny, nx = xi.shape
        xi, yi = xi.flatten(), yi.flatten()
        dist = self.distance_matrix(self.x, self.y, xi, yi)
        weights = 1.0 / dist
        weights /= weights.sum(axis=0)
        zi = np.dot(weights.T, self.z)
        return zi.reshape((ny, nx)).astype("float64")

    @staticmethod
    def distance_matrix(x0, y0, x1, y1):
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T
        d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
        d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
        return np.hypot(d0, d1)

class RegularGrid(IDW):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

    def interpolate(self, xi, yi, method="idw"):
        if method == "idw":
            zi = self.idw_interpolate(xi, yi)
        if method == "scipy":
            zi = self.scipy_interpolate(xi, yi)
        if method == "verde":
            zi = self.verde_interpolate(xi, yi)
        return zi

    def idw_interpolate(self, xi, yi):
        zi = super().interpolate(xi, yi)
        return zi

    def scipy_interpolate(self, xi, yi, method="linear"):
        nx, ny = np.meshgrid(xi, yi)
        zi = griddata((self.x, self.y), self.z, (nx, ny), method=method)
        return zi

    def verde_interpolate(self, xi, yi, method="cubic"):
        spacing = xi[1] - xi[0]
        region = (np.min(xi), np.max(xi), np.min(yi), np.max(yi))
        coordinates = (self.x, self.y)
        projection = pyproj.Proj(proj="latlon")
        proj_coordinates = projection(*coordinates)

        grd = vd.ScipyGridder(method=method).fit(proj_coordinates, self.z)
        grid = grd.grid(
            region=region,
            spacing=spacing,
            projection=projection,
            dims=["Latitude", "Longitude"],
            data_names=["mol"],
        )
        return grid["mol"].values
