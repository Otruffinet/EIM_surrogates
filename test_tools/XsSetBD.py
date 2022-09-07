import numpy as np
import pickle
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import time
import dask
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from progress.bar import Bar
import tensorly
from methods.tensor_tools import multi_mode_dot_bd, unfold_bd
from tensorly.tucker_tensor import tucker_to_tensor

class XS_set_bd:

    def __init__( self, xs_files_list=[], keys_files_list=[], cc_files_list=[],
                  xs_shape=None, chunk_size=(5000, 4704), persist_macros=False):

        self.xs_files_list = xs_files_list
        self.keys_files_list = keys_files_list
        self.cc_files_list = cc_files_list
        self.assemblies = []
        macro_datasets, micro_datasets = [], []
        self.cc_frames = []
        self.macro_keys_dict, self.micro_keys_dict = {}, {}
        macro_chunk_size = None
        micro_chunk_size = None

        if type(xs_files_list) not in [list, tuple, np.ndarray]:
            xs_files_list = [xs_files_list]
        for filename in xs_files_list:
            file = h5py.File(filename, mode='r', rdcc_nbytes=1024 ** 2 * 6000, rdcc_nslots=1e7)
            macro_dset, micro_dset = file['macro'], file['micro']

            if macro_dset.shape[0] <= chunk_size[0]:   # avoids bugs due to the absence of chunks
                macro_chunk_size = (macro_dset.shape[0] // 2, chunk_size[1])
            else:
                macro_chunk_size = chunk_size
            macro_datasets.append(da.from_array(macro_dset, chunks=macro_chunk_size))

            if micro_dset.shape[0] <= chunk_size[0]: # avoids bugs due to the absence of chunks
                micro_chunk_size = ( micro_dset.shape[0] // 2, chunk_size[1])
            else:
                micro_chunk_size = chunk_size
            micro_datasets.append(da.from_array(micro_dset, chunks=micro_chunk_size))

        if type(keys_files_list) not in [list, tuple, np.ndarray]:
            keys_files_list = [keys_files_list]
        for filename in keys_files_list:
            dico = pickle.load(open(filename, 'rb'))
            ass_name, macro_keys, micro_keys = dico['data_name'].split('_')[0], dico['macro_keys'], dico[
                'micro_keys']
            self.assemblies.append(ass_name)
            self.macro_keys_dict[ass_name], self.micro_keys_dict[ass_name] = macro_keys, micro_keys

        if type(cc_files_list) not in [list, tuple, np.ndarray]:
            cc_files_list = [cc_files_list]
        for filename in cc_files_list:
            self.cc_frames.append(pd.read_csv(filename, index_col=0))

        if len(self.macro_keys_dict) > 0:
            self.macro_keys = [str(assembly) + '_' + key for assembly in self.assemblies for key in
                               self.macro_keys_dict[assembly]]
        if len(self.micro_keys_dict) > 0:
            self.micro_keys = [str(assembly) + '_' + key for assembly in self.assemblies for key in
                                  self.micro_keys_dict[assembly]]
        if len(self.cc_frames) > 0:
            self.concs = pd.concat(self.cc_frames, axis=0)

        if persist_macros:
            self.macro_data = self.macro_data.compute()
        self.persist_macros = persist_macros

        self.normalized_micro_data = None
        self.tensor_data = None
        self.normalize = None
        self.center = None
        self.norms = {}
        self.xs_shape = xs_shape
        self.macro_chunk_size = macro_chunk_size
        self.micro_chunk_size = micro_chunk_size
        dask.config.set(**{'array.slicing.split_large_chunks': True})


    def add_data( self, xs_filename, keys_filename=None, cc_filename=None ):
        self.xs_files_list.append(xs_filename)
        file = h5py.File(xs_filename, mode='r', rdcc_nbytes=1024 ** 2 * 6000, rdcc_nslots=1e7)

        macro_dset, micro_dset = file['macro'], file['micro']
        self.macro_data = da.concatenate([self.macro_data, da.from_array(macro_dset, chunks=self.macro_chunk_size)],
                                         axis=0)
        self.micro_data = da.concatenate([self.micro_data, da.from_array(micro_dset, chunks=self.micro_chunk_size)],
                                         axis=0)

        dico = pickle.load(open(keys_filename, 'rb'))
        ass_name, macro_keys, micro_keys = dico['data_name'].split('_')[0], dico['macro_keys'], dico['micro_keys']
        self.assemblies.append(ass_name)  # à supprimer à l'avenir
        self.macro_keys[ass_name], self.micro_keys[ass_name] = macro_keys, micro_keys

        new_concs = pd.read_csv(cc_filename, index_col=0)
        self.concs = pd.concat([self.concs, new_concs], axis=0)


    def normalize_data( self, normalize, center=None, compute_norms=True ):
        self.norms = {}  # In case of several successive normalization attempts
        data = self.micro_data
        start = time.time()

        ## Centering
        if center == 'rows':
            self.norms['h_mean'] = data.mean(axis=1)[:, None]
            if compute_norms:
                self.norms['h_mean'] = self.norms['h_mean'].compute()
            data = data - self.norms['h_mean']

        elif center == 'columns':
            self.norms['v_mean'] = data.mean(axis=0)[None, :]
            if compute_norms:
                self.norms['v_mean'] = self.norms['v_mean'].compute()
            data = data - self.norms['v_mean']

        ## Normalisation
        if normalize == 'max':
            self.norms['norm'] = da.max(da.fabs(data), axis=1)[:, None]
        elif normalize == 'std':
            self.norms['norm'] = da.std(data, axis=1)[:, None]
        elif normalize == 'std_cols':
            self.norms['norm'] = da.std(data, axis=0)[None, :]
        elif normalize == 'log_div':
            self.norms['norm'] = da.log(da.max(da.fabs(data), axis=1) + 1)[:, None]
        elif normalize == 'sqrt':
            self.norms['norm'] = da.sqrt(da.max(da.fabs(data), axis=1))[:, None]
        elif normalize == 'no_norm':
            self.norms['norm'] = da.ones(data.shape[0])
        else:
            print('Unrecognized normalization name. No normalisation is performed')
            normalized_data = data

        if compute_norms:
            self.norms = dask.compute(self.norms)[0]

        if normalize in ['max', 'std', 'sqrt', 'log_div', 'std_cols']:
            normalized_data = data / self.norms['norm']
        elif normalize == 'no_norm':
            normalized_data = data

        self.normalized_micro_data = normalized_data
        self.normalize = normalize
        self.center = center
        end = time.time()
        print('Data is normalized. Time elapsed : {0} s'.format(end - start))

    def renormalize_data( self, data, norms, after_flattening=False):
        data = data.copy()

        if after_flattening:
            ## This is a procedure for the handling of tensor data : the order of cross-sections can change after their storage in a tensor.
            ## We thus need to match correctly the sections and their stored norms
            ## ! This method is not compatible with column-wise normalization !

            void_value = -999999  # For an integer array, the void value cannot be NaN
            new_indices = np.full(self.tensor_data.shape[:-1], void_value, dtype=int)
            for i, coords in enumerate(self.key_idx_tens.T):
                new_indices[coords[0], coords[1]] = i  # !! Only effective for 3D tensors at the moment !
            new_indices = new_indices.flatten()
            reshuffled_norms = {}
            for norm_name, norm_element in norms.items():
                reshuffled_norms[norm_name] = np.zeros((len(new_indices), 1))
                for j, index in enumerate(new_indices):
                    if index != void_value:
                        reshuffled_norms[norm_name][j] = norm_element[index]
            norms = reshuffled_norms

        if 'norm' in norms:
            data = data * norms['norm']
        if 'h_mean' in norms and 'min' not in norms:
            data = data + norms['h_mean']
        if 'v_mean' in norms:
            data = data + norms['v_mean']

        if after_flattening:
            return data, new_indices   # The indices make it possible to keep track of the new order of sections
        return data

    ##----------------------------------------------------------------------------------------------------------------------
    def compute_macros( self, micro_data_rec, out_of_core=True, compute=False, procedure='fast', shuffled_micro_keys=None):
        ##  shuffled_micro_keys is only useful for the robust procedure, to keep track of the order of sections

        if (not out_of_core) or procedure=='fast':
            xx = np
            compute = False
        else:
            xx = da

        if procedure=='fast':
            if self.xs_shape is None:
                raise ValueError("\n Error: calculation of macro errors requires to provide the shape of an xs !")
            elif np.product(self.xs_shape) != self.micro_data.shape[1]:
                raise ValueError("\n Error: provided xs shape not matching this of the compressed array !")

            macro_data_rec = np.zeros(self.macro_data.shape)
            current_div = 0
            current_macro_div = 0
            for i, assembly in enumerate(self.assemblies):
                macro_keys, micro_keys = self.macro_keys_dict[assembly], self.micro_keys_dict[assembly]
                # !! These keys don't contain the name of the assembly !!
                concs = self.cc_frames[i]
                cc_keys = concs.index.tolist()
                cc_idx_dict = {cc_key: idx for idx, cc_key in enumerate(cc_keys)}
                xs_keys_split = [key.split('_') for key in micro_keys]

                divisions = [0]
                count = 0
                macro_count = 0
                skip = np.zeros(2 * len(macro_keys))
                # Some expected macro sections can be missing when some concentrations are zero. Consequently, we have to
                # leave some slack in the final format, hence the factor 2
                words = xs_keys_split[0]
                previous_rea = words[2] + '_' + words[3] + '_' + words[4] + '_' + words[5]
                previous_med = words[0]
                cc_idx_rep = [None] * len(micro_keys)
                for j, words in enumerate(xs_keys_split):
                    cc_key, medium, reaction = assembly + '_' + words[0] + '_' + words[1], words[0], words[2] + '_' + \
                                               words[3] + '_' + words[4] + '_' + words[5]
                    cc_idx_rep[j] = cc_idx_dict.get(cc_key)
                    if reaction != previous_rea or medium != previous_med:
                        # The keys (medium-isotope-reaction-group-group_out_anisotropy) are stored by contiguous medium and isotope ;
                        # corresponding sections of a contiguous block will be summed together in a same macro-section.
                        # If medium or isotope has changed from previous to current iteration, it means that a new block has been reached
                        divisions.append(j)

                        macro_key = previous_med + '_macro_' + previous_rea
                        if macro_key != macro_keys[macro_count]:
                            # Case where a macro section was expected to exist but is missing. We must keep track of this
                            skip[count] = True
                        else:
                            macro_count += 1
                        previous_rea = reaction
                        previous_med = medium
                        count += 1

                divisions.append(len(micro_keys))
                cc_mat_rep = concs.values[cc_idx_rep]  # Concentrations matching the current (ordered) sections are fetched
                micro_data = micro_data_rec[current_div : current_div + len(micro_keys), :]
                if not isinstance(micro_data, np.ndarray):
                    micro_data = micro_data.compute()

                n_skipped = 0
                for j in range(len(divisions) - 1):
                    if skip[j]:
                        # Procedure to account for the missing sections
                        n_skipped += 1
                        continue
                    part_data, part_cc = micro_data[divisions[j]:divisions[j + 1], :], \
                                         cc_mat_rep[divisions[j]:divisions[j + 1], :][:, :, None, None, None]
                    # Concentrations only depend on the burnup, therefore several shape changes have to be made to multiply
                    # them correctly with the sections
                    old_shape, new_shape = part_data.shape, tuple([len(part_data)] + list(self.xs_shape))
                    contributions = np.multiply(part_data.reshape(new_shape), part_cc).reshape(old_shape)
                    # A contribution is of the form "section * concentration"
                    macro_data_rec[current_macro_div + j - n_skipped] = np.sum(contributions, axis=0)

                current_div += len(micro_keys)
                current_macro_div += len(macro_keys)

        elif procedure=='robust':
            if shuffled_micro_keys is not None:
                micro_keys = shuffled_micro_keys
            else:
                micro_keys = self.micro_keys
            macro_keys = self.macro_keys
            macro_data_rec = xx.zeros_like(self.macro_data)
            cc_idx_rep, macro_idx_rep = self.compute_macro_indices(micro_keys=micro_keys)
            # Fetches the indices of concentrations and macro-sections matching micro-sections in the given order

            if out_of_core:
                cc_mat = da.from_array(self.concs.values, chunks=micro_data_rec.chunksize)
            else:
                cc_mat = self.concs.values
            # Construction of the matrix of concentrations matching the matrix of sections
            cc_mat_rep = cc_mat[cc_idx_rep][:,:,None, None, None]

            old_shape, new_shape = micro_data_rec.shape, tuple([len(micro_data_rec)] + list(self.xs_shape))
            # Concentrations only depend on the burnup, therefore several shape changes have to be made to multiply
            # them correctly with the sections
            contributions = xx.multiply(micro_data_rec.reshape(new_shape), cc_mat_rep).reshape(old_shape)
            for index in range(len(macro_keys)):
                idxs_of_xs_that_contribute = np.where(macro_idx_rep == index)[0].tolist()
                macro_data_rec[index] = xx.sum(contributions[idxs_of_xs_that_contribute],axis=0)

        if compute:
            macro_data_rec = macro_data_rec.compute()
        return macro_data_rec


    def compute_macro_indices( self, micro_keys=None):
        # specifying micro_keys is only useful when working with tensors : the order of micro sections can vary according to context
        if micro_keys is None:
            micro_keys = self.micro_keys
        macro_keys = self.macro_keys
        cc_keys = self.concs.index.tolist()

        # Order of the words in a section key : assembly, medium, isotope, reaction, group, group_out, anisotropy
        xs_keys_split = [key.split('_') for key in micro_keys]
        ## Construction of the list of concentrations matching micro-sections
        cc_keys_rep = [words[0] + '_' + words[1] + '_' + words[2] for words in xs_keys_split]
        cc_idx_dict = {cc_key: idx for idx, cc_key in enumerate(cc_keys)}
        cc_idx_rep = np.array([cc_idx_dict.get(key) for key in cc_keys_rep])

        macro_keys_rep = [key.replace(xs_keys_split[i][2], 'macro') for i, key in enumerate(micro_keys)]
        ## Construction of a replica of micro_keys where each isotope name is replaced by "macro",
        ## in the aim of summing micro-sections corresponding to a same macro-section
        macro_idx_dict = {key: idx for idx, key in enumerate(macro_keys)}
        macro_idx_rep = np.array([macro_idx_dict.get(key) for key in macro_keys_rep])
        return cc_idx_rep, macro_idx_rep

    def compute_metrics( self, errs, data_for_means, hist_bins, quantiles, out_of_core, treshold=50.):
        if out_of_core:
            f1 = lambda x: da.where(x > treshold)
            err_max, err_mean, RMSE, quantiles, max_xs_norm, hist, means, num = dask.compute(
                errs.max(),
                errs.mean(),
                da.sqrt(da.mean(da.square(errs))),
                da.percentile(da.mean(errs, axis=1), quantiles),
                da.linalg.norm(errs, axis=1).max(),
                da.histogram(errs, bins=hist_bins, density=False),
                da.mean(da.fabs(data_for_means), axis=1),
                f1(errs)
            )
        else:
            f1 = lambda x: np.where(x > treshold)
            err_max, err_mean, RMSE, quantiles, max_xs_norm, hist, means, num = errs.max(), \
                                                                                         errs.mean(), \
                                                                                         np.sqrt(
                                                                                             np.mean(np.square(errs))), \
                                                                                         np.percentile(
                                                                                             np.mean(errs, axis=1),
                                                                                             quantiles), \
                                                                                         np.linalg.norm(errs,
                                                                                                        axis=1).max(), \
                                                                                         np.histogram(errs,
                                                                                                      bins=hist_bins,
                                                                                                      density=False), \
                                                                                         np.mean(np.abs(data_for_means),
                                                                                                 axis=1), \
                                                                                         f1(errs)
        metrics = {'err_max': err_max, 'err_mean': err_mean, 'RMSE': RMSE,
                   'quantiles': quantiles, 'max_xs_norm': max_xs_norm / data_for_means.shape[1],
                   'num_tresh': len(num[0])}

        return metrics, hist, means


    def compute_kinf_dict( self, output_file=None ):
        temp = {}
        res = {}
        for i, key in enumerate(self.macro_keys):
            ass, med, iso, rea, g, g_out, ani = key.split('_')
            env = ass + '_' + med
            if env not in temp:
                temp[env] = {'S_a1': 0., 'S_a2': 0., 'S_f1': 0., 'S_f2': 0., 'S_s12': 0., 'S_s21': 0.}
            if rea == 'Absorption':
                if g == '1':
                    temp[env]['S_a1'] += self.macro_data[i, :]
                elif g == '2':
                    temp[env]['S_a2'] += self.macro_data[i, :]
            elif rea == 'NuFission':
                if g == '1':
                    temp[env]['S_f1'] += self.macro_data[i, :]
                elif g == '2':
                    temp[env]['S_f2'] += self.macro_data[i, :]
            elif rea == 'Scattering':
                if (g == '1') & (g_out == '2'):
                    temp[env]['S_s12'] += self.macro_data[i, :]
                elif (g == '2') & (g_out == '1'):
                    temp[env]['S_s21'] += self.macro_data[i, :]

        for env, dico in temp.items():
            SI = (dico['S_a2'] + dico['S_s21']) / dico['S_s12']
            res[env] = (dico['S_f1'] * SI + dico['S_f2']) / ((dico['S_a1'] + dico['S_s12']) * SI - dico['S_s21'])

        if output_file is not None:
            pickle.dump(res, open(output_file, 'wb'))
        return res

    ## -------------------------------------------------------------------------------------------

    def compute_compression_errors( self, model, xs_set_train=None, errs_storage_path=None,
                                    compute_macros=True, transpose=False, persist_macros=False,
                                    out_of_core=True, macro_procedure='fast', **kwargs):

        if out_of_core:
            xx = da
        else:
            xx = np

        ## Préambule
        if xs_set_train is not None:
            if transpose:
                model.add_data(xs_set_train.normalized_micro_data.T)
            else:
                model.add_data(xs_set_train.normalized_micro_data)

            if not out_of_core and not isinstance(self.normalized_micro_data, np.ndarray):
                self.normalized_micro_data = self.normalized_micro_data.compute()

            if transpose:
                model.add_data(self.normalized_micro_data.T)
            else:
                model.add_data(self.normalized_micro_data)

        if macro_procedure=='fast' or self.persist_macros: # macro-sections are treated in pure numpy in the fast procedure
            persist_macros = True
        ##-------------------------------------------------------------------------------------------

        ## Compression
        print('\n Beginning compression...')
        start = time.time()
        if xs_set_train is not None:
            model.compute_base(**kwargs)
            beta, micro_data_rec = model.interpolate(self.micro_data, base=model.base, points=model.magic_points,
                                                    Q_inv=model.Q_inv, compute=False, use_dask=out_of_core)
            ## Notice that we are interpolating the original data, not the normalized one
            del beta
            if transpose:
                micro_data_rec = micro_data_rec.T
            ## Renormalization is not performed here, because we interpolated directly the unnormalized data
        else:
            coeffs, basis = model.compress_data(data=self.micro_data, out_of_core=out_of_core, compute=False, **kwargs)
            micro_data_rec = xx.matmul(coeffs, basis)
            if transpose:
                micro_data_rec = micro_data_rec.T
            micro_data_rec = self.renormalize_data(micro_data_rec, norms=self.norms)

        ##-------------------------------------------------------------------------------------------

        ## Computation of absolute micro errors
        if out_of_core:
            micro_errs = da.fabs(self.micro_data - micro_data_rec)
            micro_errs = micro_errs.rechunk(self.micro_chunk_size)
        else:
            self.micro_data = self.micro_data.compute()
            micro_errs = np.abs(self.micro_data - micro_data_rec)

        bins = np.logspace(-7, 5, 500)
        vec_quants = np.array([50.,90.,95.])
        metrics_args = {'errs':micro_errs, 'data_for_means':self.micro_data, 'hist_bins':bins,
                        'quantiles':vec_quants, 'out_of_core':out_of_core, 'treshold':10.}
        dict_micro_errs_abs, hist_micro_abs, true_means = self.compute_metrics(micro_errs,
                                                                                   data_for_means=self.micro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=out_of_core,
                                                                                   treshold=50.)

        print('\n Micro sections, absolute error in barns :', dict_micro_errs_abs)
        ## Compression time is computed here, because for SVD compression only starts with the call to compute_metrics()
        end = time.time()
        compression_time = end - start
        print('\n End of compression ! Duration: {0} s'.format(compression_time))

        if errs_storage_path is not None:
            micro_errs.to_hdf5(errs_storage_path + '_micro.h5', '/errs')

        ## Computation of relative micro errors
        micro_errs = micro_errs / true_means[:, None] * 1e5
        bins = np.logspace(-7, 5, 500)
        dict_micro_errs_rel, hist_micro_rel, _ = self.compute_metrics(micro_errs,
                                                                                   data_for_means=self.micro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=out_of_core,
                                                                                   treshold=50.)
        dict_micro_errs_rel = {key: value for key, value in dict_micro_errs_rel.items()}
        print('\n Micro sections, relative error in pcm :', dict_micro_errs_rel)

        if not compute_macros:
            metrics = dict_micro_errs_abs, dict_micro_errs_rel, hist_micro_abs, hist_micro_rel, compression_time
            labels = 'dict_micro_errs_abs', 'dict_micro_errs_rel', 'hist_micro_abs', 'hist_micro_rel', 'compression_time'
            return dict(zip(labels, metrics))

        del micro_errs
        ##-------------------------------------------------------------------------------------------

        # Computation of macro-sections
        print('\n Beginning computation of macro sections...')
        start = time.time()
        macro_data_rec = self.compute_macros(micro_data_rec, procedure=macro_procedure, compute=persist_macros)
        end = time.time()
        macro_compute_time = end - start
        print('Computation of macro sections ended. Time spent : {0} s'.format(macro_compute_time))
        ##-------------------------------------------------------------------------------------------

        ## Computation of absolute macro errors
        print("Beginning computation of macro errors...")
        start = time.time()
        bins = np.logspace(-9, 3, 500)
        boolean = (macro_procedure != 'fast') and (not persist_macros) and out_of_core
        if boolean:
            macro_errs = da.fabs(self.macro_data - macro_data_rec)
        else:
            if not isinstance(self.macro_data, np.ndarray):
                self.macro_data = self.macro_data.compute()
            macro_errs = np.abs(self.macro_data - macro_data_rec)

        dict_macro_errs_abs, hist_macro_abs, true_means = self.compute_metrics(macro_errs,
                                                                                        data_for_means=self.macro_data,
                                                                                        hist_bins=bins,
                                                                                        quantiles=vec_quants,
                                                                                        out_of_core=boolean,
                                                                                        treshold=1.)
        print('\n Macro sections, absolute error in cm^-1 :', dict_macro_errs_abs)

        if errs_storage_path is not None:
            h5f = h5py.File(errs_storage_path + '_macro.h5', 'w', rdcc_nbytes=1024 ** 2 * 6000, rdcc_nslots=1e7)
            h5f.create_dataset('/errs', data=macro_errs)


        ## Computation of relative macro errors
        macro_errs = macro_errs / true_means[:, None] * 1e5
        bins = np.logspace(-7, 5, 500)
        dict_macro_errs_rel, hist_macro_rel, _ = self.compute_metrics(macro_errs,
                                                                                        data_for_means=self.macro_data,
                                                                                        hist_bins=bins,
                                                                                        quantiles=vec_quants,
                                                                                        out_of_core=boolean,
                                                                                        treshold=50.)
        end = time.time()
        print("Computation of macro errors finished. Duration: {0} s".format(end - start))
        dict_macro_errs_rel = {key: value for key, value in dict_macro_errs_rel.items()}
        print('\n Macro sections, relative error in pcm :', dict_macro_errs_rel)

        del macro_errs, micro_data_rec, macro_data_rec

        metrics = dict_micro_errs_abs, dict_micro_errs_rel, hist_micro_abs, hist_micro_rel, dict_macro_errs_abs, \
                  dict_macro_errs_rel, hist_macro_abs, hist_macro_rel, compression_time
        labels = 'dict_micro_errs_abs', 'dict_micro_errs_rel', 'hist_micro_abs', 'hist_micro_rel', 'dict_macro_errs_abs', \
                 'dict_macro_errs_rel', 'hist_macro_abs', 'hist_macro_rel', 'compression_time'
        return dict(zip(labels, metrics))


##----------------------------------------------------------------------------------------------------------------------

    ## Routines for tensors
    def compute_tensor_errors( self, model, ranks, out_of_core=False, compute_macros=True, persist_macros=True, 
                               macro_procedure='fast', **kwargs ):
        if self.tensor_data is None:
            raise Exception('Error : this xs set doesn\' t contain tensorized data !')
            
        model.add_data(self.tensor_data)
        if out_of_core:
            xx = da
        else:
            xx = np
            persist_macros = True

        ## Compression
        print('\n Beginning compression...')
        start = time.time()
        core, factors = model.tensor_compression(ranks, out_of_core=out_of_core, **kwargs)
        print('Core tensor shape :', core.shape)
        print('Factors shape :', [factor.shape for factor in factors])
        end = time.time()
        print('Compression ended. Time elapsed: {0} s'.format(end - start))

        ## Decompression
        print('Begin data decompression...')
        start = time.time()
        if not out_of_core:
            compressed_data = tucker_to_tensor((core, factors))
        else:
            factors = [da.from_array(matrix) for matrix in factors]
            compressed_data = multi_mode_dot_bd(da.from_array(core), factors)
        end = time.time()
        print('Decompression ended. Time elapsed: {0} s'.format(end - start))

        ## Tensor unfolding
        if self.protocol == 'media-rea-param':
            if not out_of_core:
                matrix_data = tensorly.unfold(tensorly.tensor(self.tensor_data), -1).T
                matrix_data_rec = tensorly.unfold(compressed_data, -1).T
            else:
                matrix_data = unfold_bd(self.tensor_data, -1).T
                matrix_data_rec = unfold_bd(compressed_data, -1).T
        else:
            print('Error : no protocol is specified')
            return

        ## Renormalization
        matrix_data, new_indices = self.renormalize_data(matrix_data, self.norms, after_flattening=True)
        matrix_data_rec, trash = self.renormalize_data(matrix_data_rec, self.norms, after_flattening=True)
        mask = np.where(new_indices >= 0)[0]  # Tensorization created empty lines, which must be removed

        matrix_data = matrix_data[mask, :]
        matrix_data_rec = matrix_data_rec[mask, :]
        
        micro_errs = xx.fabs(matrix_data - matrix_data_rec)
        self.micro_data = matrix_data

        ##---------------------------------------------------------------------------------------------------

        ## Computation of absolute micro errors
        bins = np.logspace(-9, 3, 500)
        vec_quants = np.array([50.,90.,95.])
        if out_of_core:
            micro_errs.compute_chunk_sizes()  # !! Nécessaire à effectuer la division par les moyennes !
        dict_micro_errs_abs, hist_micro_abs, true_means = self.compute_metrics(micro_errs,
                                                                                   data_for_means=self.micro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=out_of_core,
                                                                                   treshold=50.)
        print('\n Micro sections, absolute error in barns :', dict_micro_errs_abs)
        end = time.time()
        compression_time = end - start
        print('\n End of compression ! Duration: {0} s'.format(compression_time))
        
        ## Computation of relative micro errors
        micro_errs = micro_errs / true_means[:, None] * 1e5
        bins = np.logspace(-5, 5, 500)
        dict_micro_errs_rel, hist_micro_rel, _ = self.compute_metrics(micro_errs,
                                                                                   data_for_means=self.micro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=out_of_core,
                                                                                   treshold=50.)
        dict_micro_errs_rel = {key: value for key, value in dict_micro_errs_rel.items()}
        print('\n Micro sections, relative error in pcm :', dict_micro_errs_rel)
        if not compute_macros:
            metrics = dict_micro_errs_abs, dict_micro_errs_rel, hist_micro_abs, hist_micro_rel, compression_time
            labels = 'dict_micro_errs_abs', 'dict_micro_errs_rel', 'hist_micro_abs', 'hist_micro_rel', 'compression_time'
            return dict(zip(labels, metrics))
        ##---------------------------------------------------------------------------------------------------

        ## Macro sections computation
        print('\n Beginning computation of macro sections...')
        start = time.time()
        ## Something should be done for the robust case
        macro_data_rec = self.compute_macros(matrix_data_rec, out_of_core=out_of_core, compute=persist_macros, 
                                             procedure=macro_procedure)
        end = time.time()
        macro_compute_time = end - start
        print('Computation of macro sections ended. Time spent : {0} s'.format(macro_compute_time))
        ##---------------------------------------------------------------------------------------------------

        ## Absolute macro errors computation
        print("Beginning computation of macro errors...")
        start = time.time()
        macro_errs = xx.fabs(self.macro_data - macro_data_rec)
        bins = np.logspace(-12, -2, 500)
        boolean = (macro_procedure != 'fast') and (not persist_macros) and out_of_core
        dict_macro_errs_abs, hist_macro_abs, true_means = self.compute_metrics(macro_errs,
                                                                                   data_for_means=self.macro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=boolean,
                                                                                   treshold=50.)
        print('\n Macro sections, absolute error in cm^-1 :', dict_macro_errs_abs)

        ## Relative macro errors computation
        macro_errs = macro_errs / true_means[:, None] * 1e5
        bins = np.logspace(-5, 5, 500)
        dict_macro_errs_rel, hist_macro_rel, _ = self.compute_metrics(macro_errs,
                                                                                   data_for_means=self.macro_data,
                                                                                   hist_bins=bins,
                                                                                   quantiles=vec_quants,
                                                                                   out_of_core=boolean,
                                                                                   treshold=50.)
        end = time.time()
        print("Computation of macro errors finished. Duration: {0} s".format(end - start))
        dict_macro_errs_rel = {key: value for key, value in dict_macro_errs_rel.items()}
        print('\n Macro sections, relative error in pcm :', dict_macro_errs_rel)

        metrics = dict_micro_errs_abs, dict_micro_errs_rel, hist_micro_abs, hist_micro_rel, dict_macro_errs_abs, \
                  dict_macro_errs_rel, hist_macro_abs, hist_macro_rel, compression_time
        labels = 'dict_micro_errs_abs', 'dict_micro_errs_rel', 'hist_micro_abs', 'hist_micro_rel', 'dict_macro_errs_abs', \
                 'dict_macro_errs_rel', 'hist_macro_abs', 'hist_macro_rel', 'compression_time'
        return dict(zip(labels, metrics))

    ##------------------------------------------------------------------------------------------------------------------------------

    def save_compressed_data( self, model, filename, xs_shape, var_names, **kwargs ):
        model.add_data(self.normalized_micro_data)
        coeffs, base = model.compress_data(data=self.micro_data, compute=True, norms=self.norms,**kwargs)
        coeffs = self.renormalize_data(coeffs, norms=self.norms)
        # Renormalizing coefficients assumes that normalization has been performed row-wise
        coeffs_file = h5py.File(filename + '_coeffs.h5', 'w')
        base_file = h5py.File(filename + '_base.h5', 'w')

        ## Storage of coefficients
        xs_keys = np.array(self.micro_keys)
        permutation = np.argsort(xs_keys)
        coeffs, xs_keys = coeffs[permutation], xs_keys[permutation]
        previous_ass, previous_med, iso, rea, g, g_out, ani = xs_keys[0].split('_')
        previous_group = coeffs_file.create_group(previous_ass)
        previous_div = 0
        for i, key in enumerate(xs_keys):
            assembly, medium, iso, rea, g, g_out, ani = key.split('_')
            if medium != previous_med:
                # If medium has changed between previous and current iteration, it means that we moved to the next chunk
                # and we can store the previous one
                previous_group.create_dataset(previous_med, data=coeffs[previous_div:i, :], compression='gzip',
                                                     compression_opts=0, chunks=True)
                previous_div = i
                previous_med = medium
            if assembly != previous_ass:
                ## If the assembly has changed, we move to another storage group
                previous_ass = assembly
                previous_group = coeffs_file.create_group(previous_ass)

        ## Storage of the basis
        base = base.reshape((base.shape[0], xs_shape[0], xs_shape[1], xs_shape[2], xs_shape[3]))
        for i in range(xs_shape[0]):
            group = base_file.create_group(var_names[0] + '_' + str(i))
            for j in range(xs_shape[1]):
                subgroup = group.create_group(var_names[1] + '_' + str(j))
                for k in range(xs_shape[2]):
                    subsubgroup = subgroup.create_group(var_names[2] + '_' + str(k))
                    for l in range(xs_shape[3]):
                        dset = subsubgroup.create_dataset(var_names[3] + '_' + str(l), data=base[:, i, j, k, l],
                                                          compression='gzip', compression_opts=0)


    def check_data( self, assembly, medium, point_indices, xs_shape ):
        xs_keys = np.array(self.micro_keys)
        col_idx = np.ravel_multi_index(point_indices, xs_shape)
        slice = self.micro_data[:, col_idx].compute()
        permutation = np.argsort(xs_keys)
        xs_values, xs_keys = slice[permutation], xs_keys[permutation]
        right_assembly_is_reached = False
        right_medium_is_reached = False
        for i, key in enumerate(xs_keys):
            current_ass, current_med, iso, rea, g, g_out, ani = key.split('_')
            if not right_assembly_is_reached and current_ass == assembly:
                right_assembly_is_reached = True
            if right_assembly_is_reached:
                if not right_medium_is_reached and current_med == medium:
                    right_medium_is_reached = True
                    index_start = i
                elif right_medium_is_reached and current_med != medium:
                    return xs_values[index_start:i]

    ##------------------------------------------------------------------------------------------------------------------------------
    def plot_2d( self, xs_values, param_1_values, param_2_values, axes_labels, interpol_values=None, save=False,
                 output_name=None, just_xs=False, test_points=None, vars_test=None, title=None, z_title='xs (barn)', 
                 cmap='plasma_r', w_color='black' ):

        coords_1, coords_2 = np.meshgrid(param_2_values, param_1_values, indexing='ij')  # !! Reversed order !!

        plt.close('all')
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(coords_1, coords_2, xs_values.T, label='cross-section', color=w_color)

        if not just_xs:
            ax.plot_wireframe(coords_1, coords_2, interpol_values.T, label='interpolation', color='red')

        if test_points is not None:
            if isinstance(test_points, pd.Series): # the case of a single point must be handled separately
                ax.scatter(test_points.at[vars_test[1]], test_points.at[vars_test[0]], test_points.at['xs'], marker='o',
                           label='test_points')
            else:
                test_points = test_points.reset_index(drop=True)
                values_xs, values_1, values_2 = test_points['xs'].to_numpy(), test_points[vars_test[0]].to_numpy(), \
                                                test_points[vars_test[1]].to_numpy()
                if 'colors' in test_points.columns:
                    colors = test_points['colors'].to_numpy()
                    ax.scatter(values_2, values_1, values_xs, c=colors, marker='o', label='test_points', cmap=cmap)
                else:
                    ax.scatter(values_2, values_1, values_xs, marker='o', label='test_points')

        ax.set_xlabel(axes_labels[1])
        ax.set_ylabel(axes_labels[0])
        ax.set_zlabel(z_title)
        plt.legend()
        if title is not None:
            plt.title(title)
        if save:
            plt.savefig(output_name)
        else:
            plt.show()
        plt.close('all')

    def plot_1d( self, xs_values, param_values, ax_label, inter=None, save=False, output_name=None, just_xs=False,
                 test_points=None, var_test=None, title=None, y_title='xs (barn)', cmap='plasma_r' ):
        plt.close('all')
        plt.figure()
        ax = plt.axes()
        if not just_xs:
            ax.plot(param_values, inter, label='interpolation')
        ax.plot(param_values, xs_values, label='cross-section')

        if test_points is not None:
            if isinstance(test_points, pd.Series):  # the case of a single point must be handled separately
                ax.scatter(test_points.at[var_test], test_points.at['xs'], marker='o', label='test_points')
            else:
                test_points = test_points.reset_index(drop=True)
                values_xs, values_p = test_points['xs'].to_numpy(), test_points[var_test].to_numpy()
                if 'colors' in test_points.columns:
                    colors = test_points['colors'].to_numpy()
                    ax.scatter(values_p, values_xs, c=colors, marker='o', label='test_points', cmap=cmap)
                else:
                    ax.scatter(values_p, values_xs, marker='o', label='test_points')

        plt.legend()
        plt.xlabel(ax_label)
        plt.ylabel(y_title)
        if title is not None:
            plt.title(title)
        if save:
            plt.savefig(output_name)
        else:
            plt.show()
        plt.close('all')
        
    def plot_xs( self, key, par_dict, save=False ):
        assembly, medium, iso, rea, g, g_out, ani = key.split('_')
        if iso=='macro':
            key_idx_dict = {key: idx for idx, key in enumerate(self.macro_keys)}
            pos = key_idx_dict.get(key)
            if pos is None:
                raise IndexError('Error : key absent from xs index !')
            xs_tensor = self.macro_data[pos, :].compute().reshape(self.xs_shape)
        else:
            key_idx_dict = {key: idx for idx, key in enumerate(self.micro_keys)}
            pos = key_idx_dict.get(key)
            if pos is None:
                raise IndexError('Error : key absent from xs index !')
            xs_tensor = self.micro_data[pos, :].compute().reshape(self.xs_shape)

        title = 'Section :' + key
        idx_tf, idx_tm, idx_cb, idx_bu = len(par_dict['Tf']) // 2, \
                                         len(par_dict['Tm']) // 2, \
                                         len(par_dict['Cb']) // 2, \
                                         len(par_dict['Bu']) // 2

        self.plot_1d(xs_tensor[:, idx_tf, idx_tm, idx_cb], par_dict['Bu'], 'Bu (MWj/t)', just_xs=True, save=save,
                output_name='Figures/Sections/' + key + 'bu.png', title=title)
        self.plot_1d(xs_tensor[idx_bu, :, idx_tm, idx_cb], par_dict['Tf'], 'Tf (K)', just_xs=True, save=save,
                output_name='Figures/Sections/' + key + 'tf.png', title=title)
        self.plot_1d(xs_tensor[idx_bu, idx_tf, :, idx_cb], par_dict['Tm'], 'Tm (K)', just_xs=True, save=save,
                output_name='Figures/Sections/' + key + 'tm.png', title=title)
        self.plot_1d(xs_tensor[idx_bu, idx_tf, idx_tm, :], par_dict['Cb'], 'Cb (ppm)', just_xs=True, save=save,
                output_name='Figures/Sections/' + key + 'tf.png', title=title)

        self.plot_2d(xs_tensor[:, :, idx_tm, idx_cb], par_dict['Bu'], par_dict['Tf'], ('Bu (MWj/t)', 'Tf (K)'), just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'bu_tf.png', title=title, cmap='plasma_r')
        self.plot_2d(xs_tensor[:, idx_tf, :, idx_cb], par_dict['Bu'], par_dict['Tm'], ('Bu (MWj/t)', 'Tm (K)'), just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'bu_tm.png', title=title, cmap='plasma_r')
        self.plot_2d(xs_tensor[:, idx_tf, idx_tm, :], par_dict['Bu'], par_dict['Cb'], ('Bu (MWj/t)', 'Cb (ppm)'),
                just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'bu_cb.png', title=title, cmap='plasma_r')
        self.plot_2d(xs_tensor[idx_bu, :, :, idx_cb], par_dict['Tf'], par_dict['Tm'], ('Tf (K)', 'Tm (K)'), just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'tf_tm.png', title=title, cmap='plasma_r')
        self.plot_2d(xs_tensor[idx_bu, :, idx_tm, :], par_dict['Tf'], par_dict['Cb'], ('Tf (K)', 'Cb (ppm)'), just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'tf_cb.png', title=title, cmap='plasma_r')
        self.plot_2d(xs_tensor[idx_bu, idx_tf, :, :], par_dict['Tm'], par_dict['Cb'], ('Tm (K)', 'Cb (ppm)'), just_xs=True,
                save=save, output_name='Figures/Sections/' + key + 'tm_cb.png', title=title, cmap='plasma_r')

    def plot_cc( self, key, par_dict, save=False ):
        cc_vec = self.concs.loc[key, :].values
        title = 'Section :' + key
        idx_tf, idx_tm, idx_cb, idx_bu = len(par_dict['Tf']) // 2, \
                                         len(par_dict['Tm']) // 2, \
                                         len(par_dict['Cb']) // 2, \
                                         len(par_dict['Bu']) // 2

        self.plot_1d(cc_vec, par_dict['Bu'], 'Bu (MWj/t)', just_xs=True, save=save,
                output_name='Figures/Concentrations/' + key + 'bu.png', title=title,
                y_title='concentration (.10^24 cm^-3)')

    ##------------------------------------------------------------------------------------------------------------------------------

    def to_tensor( self, filename, protocol='media-rea-param', chunksize=[40, 40, 4704] ):
        if protocol == 'media-rea-param':
            xs_keys_split = [key.split('_') for key in self.micro_keys]
            media_rep, labels_rep = np.empty(len(xs_keys_split), dtype='<U20'), np.empty(len(xs_keys_split),dtype='<U40')  # Attention à ne pas dépasser !
            for i, words in enumerate(xs_keys_split):
                media_rep[i] = words[0] + '_' + words[1]
                labels_rep[i] = words[3] + '_' + words[4] + '_' + words[5] + '_' + words[6] + '_' + words[2]  # les labels sont déformés pour être rangés en ordre de calcul des macros
            media, media_indices = np.unique(media_rep, return_inverse=True)
            labels, labels_indices = np.unique(labels_rep, return_inverse=True)
            key_idx_tens = np.vstack((media_indices, labels_indices))

            chunksize[-1] = self.micro_data.shape[1]
            tensor = da.zeros((len(media), len(labels), self.micro_data.shape[1]), chunks=tuple(chunksize))
            bar = Bar('Tensor filling', max=len(self.normalized_micro_data))
            for i, xs in enumerate(self.normalized_micro_data):
                tensor[media_indices[i], labels_indices[i], :] = xs
                bar.next()
            bar.finish()
            chunks = (min(self.macro_data.shape[0], self.macro_chunk_size[0]),
                      min(self.macro_data.shape[1], self.macro_chunk_size[1]))
            metadata = {'norms': self.norms, 'protocol': protocol, 'keys_files_list': self.keys_files_list,
                        'cc_files_list': self.cc_files_list, 'key_idx_tens': key_idx_tens, 'chunk_size': chunks}
        else:
            raise NotImplementedError("Only storage format implemented for now is 'media-rea-param !")

        self.tensor_data = tensor
        da.to_zarr(tensor, filename, overwrite=True)
        pickle.dump(metadata, open(filename.replace('.zarr', '_metadata.pickle'), 'wb'))
        if isinstance(self.macro_data, np.ndarray):
            h5f = h5py.File(filename.replace('.zarr', '_macro.h5'), 'w', rdcc_nbytes=1024 ** 2 * 6000, rdcc_nslots=1e7)
            h5f.create_dataset('/macro', data=self.macro_data, chunks=chunks)
        else:
            self.macro_data.to_hdf5(filename.replace('.zarr', '_macro.h5'), '/macro')


    def from_tensors( self, file_paths, chunksize=None, in_RAM=False, persist_macros=True ):
        ## !! This method is not robust : it supposes thatt the initial sections matrix was stored in lexicographical order
        ## assembly-medium-...  !!
        self.keys_files_list = []
        self.cc_files_list = []
        key_idx_tens_list = []
        norms_dict = {}
        macro_dsets = []
        tensors = []
        tick = True
        n_indices = 0
        if type(file_paths) not in [list, tuple, np.ndarray]:
            file_paths = [file_paths]
        for path in file_paths:
            metadata = pickle.load(open(path.replace('.zarr', '_metadata.pickle'), 'rb'))
            if tick :
                self.macro_chunk_size = metadata['chunk_size']
                self.protocol = metadata['protocol']
                tick = False
            if type(metadata['keys_files_list']) not in [list, tuple, np.ndarray]:
                metadata['keys_files_list'] = [metadata['keys_files_list']]
            for key_path in metadata['keys_files_list']:
                self.keys_files_list.append(key_path)
            if type(metadata['cc_files_list']) not in [list, tuple, np.ndarray]:
                metadata['cc_files_list'] = [metadata['cc_files_list']]
            for cc_path in metadata['cc_files_list']:
                self.cc_files_list.append(cc_path)

            file = h5py.File(path.replace('.zarr', '_macro.h5'), mode='r', rdcc_nbytes=1024 ** 2 * 6000,rdcc_nslots=1e7)
            dset = list(file.values())[0]
            macro_dsets.append(da.from_array(dset, chunks=self.macro_chunk_size))
            tensor_dset = da.from_zarr(path, chunks=chunksize)
            tensors.append(tensor_dset)
            inc_n = tensor_dset.shape[0]

            indices = metadata['key_idx_tens']
            indices[0,:] += n_indices  # On incrémente les indices des media pour ranger correctement les sections dans le tenseur final
            key_idx_tens_list.append(indices)
            n_indices += inc_n
            for norm_name, values in metadata['norms'].items():
                if norm_name not in norms_dict:
                    norms_dict[norm_name] = []
                norms_dict[norm_name].append(values)

        self.macro_data = da.concatenate(macro_dsets, axis=0)
        print('Macro data :', self.macro_data)
        self.tensor_data = da.concatenate(tensors, axis=0)
        print('Tensor data : ', self.tensor_data)
        self.norms = {}
        for norm_name, values_list in norms_dict.items():
            self.norms[norm_name] = np.concatenate(values_list, axis=0)
        self.key_idx_tens = np.concatenate(key_idx_tens_list, axis=1)
        if in_RAM:
            self.tensor_data = self.tensor_data.compute()
            self.macro_data = self.macro_data.compute()
        elif persist_macros:
            self.macro_data = self.macro_data.compute()

        self.assemblies = []
        self.cc_frames = []
        for filename in self.keys_files_list:
            dico = pickle.load(open(filename, 'rb'))
            ass_name, macro_keys, micro_keys = dico['data_name'].split('_')[0], dico['macro_keys'], dico[
                'micro_keys']
            self.assemblies.append(ass_name)
            self.macro_keys_dict[ass_name], self.micro_keys_dict[ass_name] = macro_keys, micro_keys

        if type(self.cc_files_list) not in [list, tuple, np.ndarray]:
            self.cc_files_list = [self.cc_files_list]
        for filename in self.cc_files_list:
            self.cc_frames.append(pd.read_csv(filename, index_col=0))

        if len(self.macro_keys_dict) > 0:
            self.macro_keys = [str(assembly) + '_' + key for assembly in self.assemblies for key in
                               self.macro_keys_dict[assembly]]
        if len(self.micro_keys_dict) > 0:
            self.micro_keys = [str(assembly) + '_' + key for assembly in self.assemblies for key in
                                  self.micro_keys_dict[assembly]]
        if len(self.cc_frames) > 0:
            self.concs = pd.concat(self.cc_frames, axis=0)
