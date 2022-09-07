import time
# from numba import jit
import numpy as np
import h5py
import sys, os
import pickle
from concurrent.futures import ThreadPoolExecutor
import glob

# @jit(nopython=True, parallel=True)
def normalize(input_path, method='max', center=None, axis=1, input_root='', output_root='',
              rdcc_nbytes=(1024**3)*4, rdcc_nslots=1e5):
    M = list(h5py.File(input_root+input_path+'.h5', mode='r', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots).values())[0]
    norms = {}

    ## Centering 
    if center == 'rows' and normalize != 'log':
        norms['h_mean'] = M.mean(axis=1)[:, None]
        M = M - norms['h_mean']
    elif center == 'columns' and normalize != 'log':
        norms['v_mean'] = M.mean(axis=0)[None, :]
        M = M - norms['v_mean']

    ## Normalization
    if method=='max':
        norms['norm'] = np.expand_dims(np.max(np.abs(M), axis=axis), axis=axis)
    elif normalize == 'std':
        norms['norm'] = np.std(M, axis=1)[:, None]
    elif normalize == 'std_cols':
        norms['norm'] = np.std(M, axis=0)[None, :]
    elif normalize == 'log':
        norms['min'] = np.min(M, axis=1)[:, None]
    elif normalize == 'log_div':
        norms['norm'] = np.log(np.max(np.fabs(M), axis=1) + 1)[:, None]
    elif normalize == 'sqrt':
        norms['norm'] = np.sqrt(np.max(np.fabs(M), axis=1))[:, None]
    elif isinstance(method, np.ufunc):  # redundant for now, but maybe the option below should be deleted
        norms = np.expand_dims(method(M, axis=axis))
    # elif hasattr(method, '__call__'):
    #     norms = np.expand_dims(method(M, axis=axis))
    else:
        raise TypeError('Proposed normalization is not default ("max") and not a function !')

    normalized_data = M / norms

    ## Special case : log transform (centering is done after normalization)
    if center == 'rows' and normalize == 'log':
        norms['h_mean'] = normalized_data.mean(axis=1)[:, None]
        normalized_data = normalized_data - norms['h_mean']
    elif center == 'columns' and normalize == 'log':
        norms['v_mean'] = normalized_data.mean(axis=0)[None, :]
        normalized_data = normalized_data - norms['v_mean']


    output_path = input_path + '_normalized'
    normalized_data_path = output_root + output_path + '.h5'
    file = h5py.File(normalized_data_path, mode='w',
                     rdcc_nbytes=sys.getsizeof(normalized_data))  # The data is read whole, so we set cache size to its size
    file.create_dataset('data', data=normalized_data, chunks=normalized_data.shape, compression='gzip', compression_opts=0)
    pickle.dump(norms, open(output_root+output_path+'_norms.pickle', 'wb'))
    return 'Normalization succeeded for file {0}'.format(input_path)


if __name__=='__main__':
    # input_paths_list = ['a0py31e_grp02_pbp_xs','a24py26e_grp02_pbp_xs']
    input_paths_list = sorted(glob.glob('data_final/*_grp20_pbp_xs.h5'))
    size = len(input_paths_list)
    method = 'max'
    axis = 1
    input_root = 'data_new/'
    output_root = 'data_normalized/'
    # rdcc_nbytes = (1024 ** 3) * 4
    rdcc_nbytes = (1024 ** 2)
    rdcc_nslots = 1e5
    method_it, axis_it, inroot_it, outroot_it, rnb_it, rns_it = [method]*size, [axis]*size, [input_root]*size, \
                                                        [output_root]*size, [rdcc_nbytes]*size, [rdcc_nslots]*size
    start = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()-3) as executor:
        for result in executor.map(normalize, input_paths_list, method_it, axis_it, inroot_it, outroot_it, rnb_it, rns_it):
            print(result)
    end = time.time()
    print('Normalization ended. Time elapsed : {0} s'.format(end-start))





