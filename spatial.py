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
        self.coordinates = (self.x, self.y)
        self.projection = pyproj.Proj(proj="latlon")
        self.proj_coordinates = self.projection(*self.coordinates)

    def interpolate(self, xi, yi, method="idw", submethod="linear", verbose=0):
        self.spacing = xi[1] - xi[0]
        self.region = (np.min(xi), np.max(xi), np.min(yi), np.max(yi))

        if method == "idw":
            zi = self.interpolate_idw(xi, yi)
        if method == "scipy":
            zi = self.interpolate_scipy(xi, yi, method=submethod)
        if method == "verde":
            zi = self.interpolate_verde(xi, yi, method=submethod)
        if method == "spline":
            zi = self.interpolate_spline(xi, yi, verbose=verbose)
        return zi

    def interpolate_idw(self, xi, yi):
        zi = super().interpolate(xi, yi)
        return zi

    def interpolate_scipy(self, xi, yi, method="linear"):
        nx, ny = np.meshgrid(xi, yi)
        zi = griddata((self.x, self.y), self.z, (nx, ny), method=method)
        return zi

    def interpolate_spline(self, xi, yi, verbose=0):
        spline = vd.SplineCV(dampings=(1e-5, 1e-3, 1e-1), mindists=(10e3, 50e3, 100e3))
        spline.fit(self.proj_coordinates, self.z)
        grid = spline.grid(
            region=self.region,
            spacing=self.spacing,
            projection=self.projection,
            dims=["Latitude", "Longitude"],
            data_names=["mol"],
        )

        if verbose:
            # We can show the best R² score obtained in the cross-validation
            print("\nScore: {:.3f}".format(spline.scores_.max()))
            print("\nBest spline configuration:")
            print("  mindist:", spline.mindist_)
            print("  damping:", spline.damping_)
        return grid["mol"].values

    def interpolate_verde(self, xi, yi, method="cubic"):
        grd = vd.ScipyGridder(method=method).fit(self.proj_coordinates, self.z)
        grid = grd.grid(
            region=self.region,
            spacing=self.spacing,
            projection=self.projection,
            dims=["Latitude", "Longitude"],
            data_names=["mol"],
        )
        return grid["mol"].values
