import numpy as np
import time
import h5py
import os
from concurrent.futures import ThreadPoolExecutor
import glob
import resource
  
def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
##----------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------

def interpolate(f_values, base, points, f_values_red=None, just_coeffs=False ):

    if f_values_red is None:  # the user can provide only the values of f at the interpolation points (surrogate model usage), or the full row (for basis construction)
        if len(f_values.shape) == 1: # in the case where we only interpolate one row    
            f_values = f_values[None,:]
        f_values_red = f_values[:, points]  # we only keep the function values at the interpolation points

    if len(base) > 1:  # numpy can't inverse (1,)-sized matrix...
        Q_values_red = base[:, points]
        Q_inv = np.linalg.inv(Q_values_red)
    else:
        Q_inv = 1 / base[:,points]

    coeffs = np.matmul(f_values_red, Q_inv)

    if just_coeffs:  # the user has the choice to return only the decomposition coefficients, or also the approximated data
        return coeffs
    else:
        inter = np.matmul(coeffs, base)
    return coeffs, inter


def interpolation_job( data_file, base, points):
    M = list(data_file.values())[0][:]  # syntax for accessing the array (supposed to be unique) contained in the hdf5 file, whatever its name may be
    coeffs, approx = interpolate(M, base=base, points=points)
    R = M - approx
    ind = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)  # fastest way to find both row an column index of maximum
    res_max = R[ind]
    del M, R
    return res_max, ind


class EIM_model:

    def __init__(self, data, rank, stochastic=False, out_of_core=False, rdcc_nbytes=(1024**2)*4, rdcc_nslots=1e5, max_workers=os.cpu_count() - 3):
        self.data = data   # data can be a list of hdf5 paths (for out_of_core computation)
        self.rank = rank
        self.stochastic = stochastic
        self.out_of_core = out_of_core
        self.base = None
        self.points = None
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots  #cache parameters for h5py file reading (out-of-core computation)
        self.max_workers = max_workers

    def compute_base( self, mesh_shape=None, verbose=True):
        # if the support is a regular grid, the user can provide its shape in order to print the position of magic points
        if mesh_shape is not None and verbose:
            print_fancy_points = True
        else:
            print_fancy_points = False

        if self.out_of_core or self.stochastic:
            data_paths = self.data
            # for stochastic and out-of-core computation, we never load the full data in memory, hence the access to data from file objects only
            data_files = [h5py.File(path, mode='r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots) for path in data_paths]
            n_partitions = len(data_paths)
            # syntax for accessing the array (supposed to be unique) contained in the hdf5 file, whatever its name may be
            divisions = np.cumsum([len(list(file.values())[0]) for file in data_files]).tolist()
            divisions.insert(0,0)
            divisions.pop()
            if verbose:
                print('Indices of data divisions : ', divisions)
            len_support = list(data_files[0].values())[0].shape[1]

        Q = np.zeros((self.rank, len_support))
        magic_points, magic_points_print = [], []
        list_p = []
        residuals = []

        # Initialisation
        if self.out_of_core or self.stochastic:
            # In all rigor, we should work on the whole data here ; in practice, performance is very insensitive on the initialization.
            # We therefore work on the first chunk alone to spare computation in the case where the data is large
            current_rows = list(data_files[0].values())[0]
        else:
            current_rows = self.data
            len_support = current_rows.shape[1]

        ind = np.unravel_index(np.argmax(np.abs(current_rows), axis=None), current_rows.shape)
        f_max = current_rows[ind]
        p_max,x_max = ind
        q_1 = current_rows[p_max] / f_max
        Q[0, :] = q_1
        list_p.append(p_max)  # not useful for computation; just here in case you want to track which rows are picked by the algorithm
        magic_points.append(x_max)
        if print_fancy_points:
            magic_points_print.append(np.unravel_index(x_max, mesh_shape))
            print('Magic Points:', magic_points_print)
        if verbose:
            print('Magic Points indexes:', magic_points)
            print('Residuals:', np.around(residuals, decimals=4))
            print('Picked rows :', list_p)
        if self.out_of_core or self.stochastic:
            del current_rows

        # Iteration
        for i in range(1, self.rank):
            if verbose:
                print("\n Iteration :", i)

            if self.stochastic:
                rng = np.random.default_rng()
                block_id = rng.integers(n_partitions)
                if verbose:
                    print("Index of the randomly picked block for this iteration :", block_id)
                current_rows = list(data_files[block_id].values())[0]
                coeffs, approx = interpolate(current_rows, base=Q[:i, :], points=magic_points)
                R = current_rows - approx
                del current_rows
                ind = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)
                res_max = R[ind]
                p_max, x_max = ind
                q = R[p_max] / res_max
                del R

            elif self.out_of_core:
                res_max = 0.
                base_it, points_it, rnb_it, rns_it = [Q[:i,:]]*len(data_files), [magic_points]*len(data_files), \
                                                     [self.rdcc_nbytes]*len(data_files), [self.rdcc_nslots]*len(data_files)

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for j, item in enumerate(executor.map(interpolation_job, data_files, base_it, points_it, rnb_it, rns_it)):
                        residual, ind = item
                        if abs(residual) > res_max:
                            res_max, ind_max, block_id = abs(residual), ind, j    # We keep track of the largest residual over all threads

                current_rows = list(data_files[block_id].values())[0]
                p_max, x_max = ind_max
                picked_row = current_rows[p_max,:]
                coeffs, approx = interpolate(picked_row, points=magic_points, base=Q[:i, :])  # We re-compute the largest residual vector from its coordinates
                residual = picked_row - approx
                q = residual / res_max
                del current_rows

            else:  # standard EIM
                coeffs, approx = interpolate(self.data, points=magic_points, base=Q[:i,:])
                R = self.data - approx
                ind = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)
                res_max = R[ind]
                p_max, x_max = ind
                q = R[p_max] / res_max

            if x_max in magic_points:  # picking the same point twice indicates grave overfitting and breaks the algorithm
                print('Warning: The same magic point has been selected twice. Exiting algorithm at iteration ' + str(i))
                return Q[:i,:], magic_points, residuals

            if self.stochastic or self.out_of_core:
                shift = divisions[block_id]
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

    def compute_errors( self, norms=None ):
        pass


if __name__=='__main__':
    # data_paths = ['results_github/a0py31e_grp02_pbp_xs_normalized.h5','results_github/a24py26e_grp02_pbp_xs_normalized.h5']
    data_paths = sorted(glob.glob('data_new/*_grp20_pbp_xs.h5'))
    model = EIM_model(data_paths, 30, stochastic=False, out_of_core=True)
    # limit_memory(64*(1024**3))
    start = time.time()
    Q, magic_points, residuals = model.compute_base(mesh_shape=(32,7,7,3), verbose=True)
    end = time.time()
    print('Base computation ended. Time elapsed : {0} s'.format(end-start))
    data1 = list(h5py.File(data_paths[0], mode='r', rdcc_nbytes=model.rdcc_nbytes, rdcc_nslots=model.rdcc_nslots).values())[0][:]
    data2 = list(h5py.File(data_paths[1], mode='r', rdcc_nbytes=model.rdcc_nbytes, rdcc_nslots=model.rdcc_nslots).values())[0][:]
    data = np.vstack((data1, data2))
    coeffs, data_rec = interpolate(data, base=Q, points=magic_points)
    err = np.abs((data-data_rec)/data.mean(axis=1)[:,None])
    print(err.max(), err.mean())






