import numpy as np
import time
import h5py
import os
import dask
import dask.array as da
import glob
##----------------------------------------------------------------------------------------------------------------

def interpolate(f_values, base, points, just_coeffs=False, f_values_red=None, compute=True, use_dask=True):
    # the 'compute' argument decides whether numerical values are returned, or only a dask object (lazy evaluation)
    if use_dask:
        xx = da
    else:
        xx = np

    if f_values_red is None:  # the user can provide only the values of f at the interpolation points (surrogate model usage), or the full row (for base construction)
        if len(f_values.shape) == 1:    # In the case where we only interpolate one row
            f_values = f_values[None,:]
        f_values_red = f_values[:, points] # we only keep the function values at the interpolation points
            
    if len(base) > 1:  # numpy can't inverse (1,)-sized matrix...
        Q_values_red = base[:, points]
        Q_inv = np.linalg.inv(Q_values_red)
    else:
        Q_inv = 1 / base[:,points]

    coeffs = xx.matmul(f_values_red, Q_inv)
    if just_coeffs:  # the user has the choice to return only the decomposition coefficients, or also the approximated data
        if use_dask and compute:
            coeffs = coeffs.compute()
        return coeffs
    else:
        inter = xx.matmul(coeffs, base)
        if use_dask and compute:
            coeffs, inter = dask.compute(coeffs, inter)
    return coeffs, inter


class EIM_bd_model:

    def __init__(self, data, rank, stochastic=False, out_of_core=False):
        self.data = data   # data can be a numpy or dask array
        self.rank = rank
        self.stochastic = stochastic
        self.out_of_core = out_of_core
        self.base = None
        self.points = None

    def compute_base( self, mesh_shape=None, verbose=True):

        if mesh_shape is not None and verbose:  # if the support is a regular grid, the user can provide its shape in order to print the position of magic points
            print_fancy_points = True
        else:
            print_fancy_points = False
            
        if (not self.out_of_core) or self.stochastic:
            xx = np
        else:
            xx = da

        data = self.data

        if self.stochastic:
            n_partitions = 0
            divisions = [0]
            current_idx = 0
            for block in data.partitions:  
            # dask arrays have a 'partitions' attributes which points to their blocks ; 
            # but there is no good way to access the index of their divisions, hence the following code.
            # Also notice that blocks are horizontal regroupments of chunks, not chunks themselves !
                n_partitions += 1  
                current_idx += block.shape[0]
                divisions.append(current_idx) 

        dask.config.set(**{'array.slicing.split_large_chunks': True}) # for performance
        Q = np.zeros((self.rank, data.shape[1]))
        magic_points, magic_points_print = [], []
        list_p = []
        residuals = []

        # Initialisation
        if self.stochastic:
            current_rows = data.partitions[0].compute()
        # In all rigor, we should work on the whole data here ; in practice, performance is very insensitive on the initialization.
        # We therefore work on the first chunk alone to spare computation
        else:
            current_rows = data

        ind = xx.unravel_index(xx.argmax(xx.fabs(current_rows), axis=None), current_rows.shape)
        f_max = current_rows[ind]
        p_max, x_max = ind
        q_1 = current_rows[p_max] / f_max
        if self.out_of_core:
            p_max, x_max, f_max, q_1 = dask.compute(p_max, x_max, f_max, q_1)
        Q[0, :] = q_1
        list_p.append(p_max)
        if print_fancy_points:
            magic_points_print.append(np.unravel_index(x_max, mesh_shape))
            print('Magic Points:', magic_points_print)
        if verbose:
            print('Magic Points indexes:', magic_points)
            print('Residuals:', np.around(residuals, decimals=4))
            print('Picked rows :', list_p)

        # Iteration
        for i in range(1, self.rank):
            if verbose:
                print("\n Iteration :", i)

            if self.stochastic:
                rng = np.random.default_rng()
                block_id = rng.integers(n_partitions)
                if verbose:
                    print("Index of the randomly picked block for this iteration :", block_id)
                current_rows = data.partitions[block_id].compute()
                shift = divisions[block_id]

            coeffs, approx = interpolate(current_rows, points=magic_points, base=Q[:i,:], compute=False, 
                                         use_dask=self.out_of_core)
            # if in stochastic mode, dask is not useful because we work directly on chunks (which are numpy arrays)
            R = current_rows - approx
            ind = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)
            res_max = R[ind]
            p_max, x_max = ind
            q = R[p_max] / res_max
            if self.out_of_core:
                ind,res_max,p_max,x_max,q = dask.compute(ind,res_max,p_max,x_max,q)

            if x_max in magic_points:  # picking the same point twice indicates grave overfitting and breaks the algorithm
                print('Warning: The same magic point has been selected twice. Exiting algorithm at iteration ' + str(i))
                return Q[:i,:], magic_points, residuals

            if self.stochastic:
                p_max += shift

            Q[i, :] = q
            list_p.append(p_max)
            residuals.append(res_max)
            magic_points.append(x_max)
            magic_points = sorted(magic_points)  # Not mandatory, but accelerates execution
            if print_fancy_points:
                magic_points_print.append(np.unravel_index(x_max, mesh_shape))
                print('Magic Points:', magic_points_print)
            if verbose:
                print('Magic Points indexes:', magic_points)
                print('Residuals:', np.around(residuals, decimals=4))
                print('Picked rows :', list_p)

        return Q, magic_points, residuals

