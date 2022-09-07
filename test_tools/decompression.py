
import numpy as np
import cProfile, pstats
from multilinear_interpolator import MultilinInterpolator
import time
import h5py
import itertools
import os

##--------------------------------------------------------------------------

## Time measurement
def io_timer():
    timing = os.times()
    return timing.elapsed - (timing.system + timing.user)

def profile_io_time(f, *args, **kwargs):
    prof = cProfile.Profile(io_timer)
    prof.runcall(f, *args, **kwargs)
    result = pstats.Stats(prof)
    result.sort_stats("time")
    result.print_stats()

def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(' %2.2f sec' % (te-ts))
        return result

    return timed

##--------------------------------------------------------------------------
## Decompression routines

def reconstruct_data(coeffs_filename, base_filename, var_values, var_names, val_point, assembly, media):
    print('Starting reconstruction...')
    start = time.time()
    to_rem = []
    interpolator = MultilinInterpolator(var_values)
    xi = interpolator._ndim_coords_from_arrays(val_point)
    indices, norm_distances, out_of_bounds, shape = interpolator._find_indices(xi)
    # print(indices, norm_distances, out_of_bounds)

    ## Interpolation des fonctions de base
    base_file = h5py.File(base_filename, mode='r')
    ## Les étapes suivantes à générer les 2**n indices des points (n nombre de variables) qui encadrent le point d'intérêt
    combinations = np.array([x for x in itertools.product([0, 1], repeat=len(var_names))])
    vertices = np.array(indices).T + combinations
    dset_shape = base_file['/'+var_names[0]+'_0'+'/'+var_names[1]+'_0'+'/'+var_names[2]+'_0'+'/'+var_names[3]+'_0'].shape  ## à remplacer par de la metadata
    base_values = np.zeros((len(vertices), dset_shape[0]))
    mid_time = time.time()
    for i,vertex in enumerate(vertices):
        path = ''
        for j in range(len(var_names)):
            path += '/'+ var_names[j] + '_' + str(vertex[j])
        base_values[i,:] = base_file[path]
    to_rem.append(time.time()-mid_time)
    reduced_var_values = [[values[indices[i][0]], values[indices[i][0]+1]] for i,values in enumerate(var_values)]
    interpolator = MultilinInterpolator(reduced_var_values)
    f_values = base_values.reshape((2,2,2,2,dset_shape[0]))
    interpolator.add_f_values(f_values)
    interpolated_base = interpolator._evaluate_linear([np.array([0]) for i in range(len(var_names))],norm_distances,out_of_bounds).squeeze()
    # print(interpolated_base, interpolated_base.shape)

    ## Lecture des coefficients
    coeffs_file = h5py.File(coeffs_filename, mode='r')
    if type(media) not in [list, tuple, np.ndarray]:
        media = [media]
    res = {}
    for medium in media:
        mid_time = time.time()
        coeffs = coeffs_file['/'+assembly+'/'+medium][()]
        to_rem.append(time.time() - mid_time)
        ## Combinaison linéaire
        res[medium] = np.matmul(coeffs,interpolated_base)
        print('Number of xs in medium {0} : {1}'.format(medium, coeffs.shape[0]))
        
    print('Truncation rank : ', coeffs.shape[1])
    end = time.time()
    print('Reconstruction ended. Time elapsed : {0} s. IO time : {1} s. Computation time : {2} s.'.format(end-start, np.sum(to_rem), end-start-np.sum(to_rem)))
    return res


def compute_macros(micro_data, concs, macro_keys, nonmacro_keys, assembly, medium): # !! Pour un assemblage seulement, et il n'y a pas le nom d'assemblage dans les clés !

    macro_data = np.zeros((len(macro_keys), micro_data.shape[1]))
    current_macro_div = 0
    cc_keys = concs.index.tolist()
    cc_idx_dict = {cc_key: idx for idx, cc_key in enumerate(cc_keys)}
    xs_keys_split = [key.split('_') for key in nonmacro_keys]

    divisions = [0]
    count = 0
    macro_count = 0
    skip = np.zeros(2*len(macro_keys))  # on n'a qu'une idée approximative de la taille de divisions
    words = xs_keys_split[0]
    previous_rea = words[2]+'_'+words[3]+'_'+words[4]+'_'+words[5]
    cc_idx_rep = [None]*len(nonmacro_keys)
    for j,words in enumerate(xs_keys_split):
        cc_key, reaction = assembly+'_'+words[0]+'_'+words[1], words[2]+'_'+words[3]+'_'+words[4]+'_'+words[5]
        cc_idx_rep[j] = cc_idx_dict.get(cc_key)
        if reaction != previous_rea: # Les clés (milieu-iso-rea-etc) sont rangées par milieu identique, puis en faisant varier l'isotope.
            # Autrement dit, toutes les sections correspondant à un même milieu-réaction sont contigues et seront sommées ensemble.
            # Si on a changé de réaction ou de medium, cela signifie qu'on aborde un nouveau bloc
            divisions.append(j)
            macro_key = medium+'_macro_'+previous_rea
            # if macro_key not in macro_keys:     # à 20 groupes, certaines sections macro sont nulles, et donc absentes de l'index.
            # Dans ce cas, il faut les sauter
            if macro_key != macro_keys[macro_count]:
                skip[count] = True
            else:
                macro_count += 1
            previous_rea = reaction
            count += 1

    divisions.append(len(nonmacro_keys))
    cc_mat_rep = concs.values[cc_idx_rep]
    n_skipped = 0
    for j in range(len(divisions)-1):
        if skip[j]:
            n_skipped += 1
            continue
        part_data, part_cc = micro_data[divisions[j]:divisions[j+1],:], cc_mat_rep[divisions[j]:divisions[j+1]]
        contributions = np.multiply(part_data, part_cc[:,None])
        # print(j,divisions[j], divisions[j+1])
        macro_data[current_macro_div+j-n_skipped] = np.sum(contributions, axis=0)
    current_macro_div += len(macro_keys)

    return macro_data

# @measure_time
def reconstruct_macro_data(coeffs_filename, base_filename, var_values, var_names, val_point, assembly, media, 
                           macro_keys, nonmacro_keys, concs):
    print('Starting reconstruction...')
    start = time.time()
    to_rem = []
    interpolator = MultilinInterpolator(var_values)
    xi = interpolator._ndim_coords_from_arrays(val_point)
    indices, norm_distances, out_of_bounds, shape = interpolator._find_indices(xi)
    # print(indices, norm_distances, out_of_bounds)

    ## Interpolation des fonctions de base
    base_file = h5py.File(base_filename, mode='r')
    ## Les étapes suivantes à générer les 2**n indices des points (n nombre de variables) qui encadrent le point d'intérêt
    combinations = np.array([x for x in itertools.product([0, 1], repeat=len(var_names))])
    vertices = np.array(indices).T + combinations
    dset_shape = base_file['/'+var_names[0]+'_0'+'/'+var_names[1]+'_0'+'/'+var_names[2]+'_0'+'/'+var_names[3]+'_0'].shape  ## à remplacer par de la metadata
    base_values = np.zeros((len(vertices), dset_shape[0]))
    mid_time = time.time()
    for i,vertex in enumerate(vertices):
        path = ''
        for j in range(len(var_names)):
            path += '/'+ var_names[j] + '_' + str(vertex[j])
        base_values[i,:] = base_file[path]
    to_rem.append(time.time()-mid_time)
    reduced_var_values = [[values[indices[i][0]], values[indices[i][0]+1]] for i,values in enumerate(var_values)]
    interpolator = MultilinInterpolator(reduced_var_values)
    f_values = base_values.reshape((2,2,2,2,dset_shape[0]))
    interpolator.add_f_values(f_values)
    interpolated_base = interpolator._evaluate_linear([np.array([0]) for i in range(len(var_names))],norm_distances,out_of_bounds).squeeze()
    # print(interpolated_base, interpolated_base.shape)

    ## Lecture des coefficients, reconstruction des macros
    coeffs_file = h5py.File(coeffs_filename, mode='r')
    if type(media) not in [list, tuple, np.ndarray]:
        media = [media]
    res = {}
    for i, medium in enumerate(media):
        mid_time = time.time()
        coeffs = coeffs_file['/'+assembly+'/'+medium][()]
        to_rem.append(time.time() - mid_time)
        macro_coeffs = compute_macros(micro_data=coeffs, concs=concs, macro_keys=macro_keys[i], nonmacro_keys=nonmacro_keys[i],
                                      assembly=assembly, medium=medium)
        ## Combinaison linéaire
        res[medium] = np.matmul(macro_coeffs,interpolated_base)
        print('Number of macro xs in medium {0} : {1}'.format(medium, macro_coeffs.shape[0]))
        
    end = time.time()
    print('Reconstruction ended. Time elapsed : {0} s. IO time : {1} s. Computation time : {2} s.'.format(end-start, np.sum(to_rem), end-start-np.sum(to_rem)))
    print('Truncation rank : ', coeffs.shape[1])
    return res





