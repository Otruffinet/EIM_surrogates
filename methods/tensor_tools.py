import numpy as np
import dask.array as da

## Big data tensor operations
def unfold_bd(tensor,mode):
    return da.reshape(da.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold_bd(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return da.moveaxis(da.reshape(unfolded_tensor, full_shape), 0, mode)

def vec_to_tensor_bd( vec, shape ):
    return da.reshape(vec, shape)

def mode_dot_bd( tensor, matrix_or_vector, mode, transpose=False):
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if matrix_or_vector.ndim == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                ))
        if transpose:
            matrix_or_vector = da.conj(da.transpose(matrix_or_vector))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False
    elif matrix_or_vector.ndim == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                ))
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            new_shape = []
        vec = True
    else:
        raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                         'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))

    res = da.dot(matrix_or_vector, unfold_bd(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return vec_to_tensor_bd(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold_bd(res, fold_mode, new_shape)

def multi_mode_dot_bd( tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False ):
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor
    res = tensor
    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue  # skip permet de multiplier tous les modes du tenseur sauf un (coeur de chaque étape d'itération de l'HOOI)
        if transpose:
            res = mode_dot_bd(res, da.conj(da.transpose(matrix_or_vec)), mode - decrement)
        else:
            res = mode_dot_bd(res, matrix_or_vec, mode - decrement)
        if matrix_or_vec.ndim == 1:
            decrement += 1
    return res

def tens_norm(tensor):
    return da.tensordot(tensor, tensor, axes=tensor.ndim)