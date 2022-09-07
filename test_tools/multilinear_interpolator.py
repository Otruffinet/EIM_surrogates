# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:24:24 2022

@author: OT266455
"""
import numpy as np
import itertools

class MultilinInterpolator:

    def __init__( self, points, bounds_error=True ):
        self.bounds_error = bounds_error

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)

        self.grid = tuple([np.asarray(p) for p in points])
        self.points = points

    def __call__( self, xi ):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        """
        ndim = len(self.grid)
        indices, norm_distances, out_of_bounds, shape = self._find_indices(xi)
        result = self._evaluate_linear(indices,
                                       norm_distances,
                                       out_of_bounds)

        return result.reshape(shape + self.values.shape[ndim:])

    def add_f_values( self, values ):
        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)
        for i, p in enumerate(self.points):
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        if len(self.points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(self.points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)
        self.values = values

    def _ndim_coords_from_arrays( self, points, ndim=None ):
        """
        Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
        """
        if isinstance(points, tuple) and len(points) == 1:
            # handle argument tuple
            points = points[0]
        if isinstance(points, tuple):
            p = np.broadcast_arrays(*points)
            n = len(p)
            for j in range(1, n):
                if p[j].shape != p[0].shape:
                    raise ValueError("coordinate arrays do not have the same shape")
            points = np.empty(p[0].shape + (len(points),), dtype=float)
            for j, item in enumerate(p):
                points[..., j] = item
        else:
            points = np.asanyarray(points)
            if points.ndim == 1:
                if ndim is None:
                    points = points.reshape(-1, 1)
                else:
                    points = points.reshape(-1, ndim)
        return points

    def _evaluate_linear( self, indices, norm_distances, out_of_bounds ):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _find_indices( self, xi ):
        ndim = len(self.grid)
        xi = self._ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        xi = xi.T
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds, xi_shape[:-1]
