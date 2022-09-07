import time
import numpy as np
import dask
import dask.array as da
import dask_ml.decomposition
import sklearn.decomposition
import tensorly
from tensorly.decomposition import tucker
from tensor_tools import unfold_bd, multi_mode_dot_bd, tens_norm


class SVD_bd:

    def __init__(self, n_components=None, data=None, algorithm='random', verbosity=0):
        self.name = 'SVD'
        self.n_components = n_components
        self.data = data
        self.algorithm = algorithm
        self.components = None
        self.directions = None
        self.explained_variance_ratio = None
        self.verbosity = verbosity

    def add_data( self, data ):
        self.data = data

    # ----------------------------------------------------------------------------------------------------------------------
    def compress_data( self, out_of_core=True, compute=True, data=None, **kwargs):
        if data is None:
            data = self.data
        if self.algorithm=='random' and out_of_core:
            u, s, v = da.linalg.svd_compressed(self.data, k=self.n_components,**kwargs)
            self.components = da.matmul(u, da.diag(s))
            self.explained_variance_ratio= s/self.data.var()
            self.directions = v
            if compute:
                self.components,self.explained_variance_ratio,self.directions = dask.compute(self.components,
                                                                                             self.explained_variance_ratio,self.directions)
            del u,s,v

        else:
            if out_of_core:
                TruncatedSVD = dask_ml.decomposition.TruncatedSVD
            else:
                TruncatedSVD = sklearn.decomposition.TruncatedSVD
            SVD = TruncatedSVD(n_components=self.n_components, **kwargs)
            X_transformed = SVD.fit_transform(self.data)
            self.explained_variance_ratio = SVD.explained_variance_ratio_
            self.directions = SVD.components_
            self.components = X_transformed
            del SVD

        return self.components, self.directions


##----------------------------------------------------------------------------------------------------------------------

class Tucker:

    def __init__(self, n_components=None, data=None, algorithm='random', verbosity=0):
        self.name = 'Tucker'
        self.n_components = n_components
        self.data = data
        self.algorithm = algorithm
        self.verbosity = verbosity

    def add_data( self, data ):
        self.data = data

    # ----------------------------------------------------------------------------------------------------------------------
    def tensor_compression( self, ranks, out_of_core=False, **kwargs):
        print('Begin HOOI computation...')
        start = time.time()
        if not out_of_core:
            data_tensor = tensorly.tensor(self.data)
            core, factors = tucker(data_tensor, rank=ranks, **kwargs)
        else:
            core, factors = self.HOOI( self.data, rank=ranks, **kwargs)
        end = time.time()
        print('HOOI computation ended. Time elapsed: {0} s'.format(end-start))
        return core, factors


    def HOOI( self, tensor, rank, n_iter_max=500, tol=10e-6, verbose=False, large_data=False, **kwargs):
        print('HOOI initialized. n_iter_max : {0}; tol : {1}'.format(n_iter_max,tol))
        modes = list(range(tensor.ndim))
        if rank is None:
            message = "No value given for 'rank'. The decomposition will preserve the original size."
            print(message)
            rank = [tensor.shape[mode] for mode in modes]
        elif isinstance(rank, int):
            message = "Given only one int for 'rank' instead of a list of {} modes. Using this rank for all modes.".format(
                len(modes))
            print(message)
            rank = tuple(rank for _ in modes)
        else:
            rank = tuple(rank)

        # SVD init
        factors = []
        for index, mode in enumerate(modes):
            Y = unfold_bd(tensor, mode)
            # print(index, Y.shape)
            if large_data:
                TruncatedSVD = dask_ml.decomposition.TruncatedSVD
                SVD = TruncatedSVD(n_components=rank[index])  # , algorithm='arpack')
                X_transformed = SVD.fit_transform(Y)
                eigenvecs = X_transformed / SVD.singular_values_
            else:
                eigenvecs, _, _ = da.linalg.svd_compressed(Y, k=rank[index], coerce_signs=True, **kwargs)
            factors.append(eigenvecs.compute())

        rec_errors = []
        norm_tensor = tens_norm(tensor)
        # factors, norm_tensor = dask.compute(factors, norm_tensor)
        norm_tensor = norm_tensor.compute()

        for iteration in range(n_iter_max):
            for index, mode in enumerate(modes):
                core_approximation = multi_mode_dot_bd(tensor, factors, modes=modes, skip=index, transpose=True)
                Y = unfold_bd(core_approximation, mode)
                # print(index, Y.shape)
                eigenvecs, _, _ = da.linalg.svd_compressed(Y, k=rank[index], coerce_signs=True, **kwargs)
                factors[index] = eigenvecs
            factors = list(dask.compute(*factors))

            core = multi_mode_dot_bd(tensor, factors, modes=modes, transpose=True)
            # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
            rec_error = da.sqrt(da.fabs(norm_tensor ** 2 - tens_norm(core) ** 2)) / norm_tensor
            rec_errors.append(rec_error.compute())

            if iteration > 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))
                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    print('converged in {} iterations.'.format(iteration))
                    break
        core = core.compute()
        return (core, factors)
