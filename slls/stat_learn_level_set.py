import os, shutil, time, warnings, pickle, h5py
import datetime, logging

import numpy as np
import multiprocessing as mp
import skfmm
from sklearn.tree import DecisionTreeRegressor

from feature_maps import feature_map_base
from init_funcs import init_func_base
from utils.data import splitter
from utils.stdout_redirector import stdout_redirector

import utils.masked_grad as mg
from neural_network import neural_network as nn


MODEL_FIT_METHODS = ('nnet', 'rf')


class stat_learn_level_set(object):
    def __init__(self, data_file, feature_map, init_func, score_func=None,
                 step='auto', band=3, rs=None): 
        """
        Initialize a statistical learning level set object.
        
        The parameters required at initialization 
        are those that can be used in *both* the training and run-time
        phases of the algorithm. Training options should be set via
        the `set_fit_options` and `set_net_options` member functions
        after initialization.

        Parameters
        ----------
        data_file: str
            This string should be of the form::
                
                some/path/to/data.h5

            where `data.h5` is formatted as specified in the 
            :meth:`slls.utils.tohdf5` module.
            `data.h5` contains the image and ground truth data.

        feature_map: feature map class
            See :class:`slls.feature_maps.feature_map_base`.

        init_func: init func class
            See :class:`slls.init_funcs.init_func_base`.

        score_func: function, default=None
            Has signature::
                
                score_func(u, seg)

            where `u` is a level set iterate and `seg` is the ground-truth
            segmentation. The default (None) uses the jaccard overlap 
            between `u > 0` and `seg`.

        step: float, default='auto'
            The step size for updating the level sets, i.e., the "delta t"
            term in the discretization of :math:`u_t = \\nu \\| Du \\|`.

            The default ('auto') determines the step size automatically 
            by using the reciprocal of the maximum ground truth speed values
            in the narrow band of the level sets in the training data at 
            the first iterate (i.e., at initialization). The CFL condition
            is satisfied by taking the maximum speed over ALL spatial 
            coordinates, but we attempt to avoid this prohibitively 
            small step size by assuming that the maximum speed observed 
            will be in the first iteration (i.e., that u0 is farthest 
            from the ground-truth).

            If 'auto', then `step` is replaced with the computed
            value after `fit` is called.

        band: int, default=3
            The "narrow band" distance.

        rs: numpy.random.RandomState, default=None
            Provide for reproducible results.
        """
        self._data_file_name = os.path.abspath(data_file)

        with self._data_file as df:
            self._ndim = df[df.keys()[0]+"/img"].ndim

        if not isinstance(feature_map, feature_map_base):
            raise ValueError(
                "`feature_map` should be a class derived from "
                "`slls.feature_maps.feature_map_base`."
            ) 

        self.feature_map = feature_map

        if not isinstance(init_func, init_func_base):
            raise ValueError(
                "`init_func` should be a class derived from "
                "`slls.init_funcs.init_func_base`."
            )

        self.init_func = init_func

        self.score_func = jaccard if score_func is None else score_func
        self.band = band

        if step != 'auto' and not isinstance(step, float):
            raise ValueError("`step` must be 'auto' or float.")
        self.step = step

        if rs is None:
            rs = np.random.RandomState()
            warnings.warn(
                "No RandomState provided. Results will not be reproducible."
            )

        self._rs = rs

        self._is_fitted    = False
        self._fit_opts_set = False
        self._net_opts_set = False

    @property
    def _data_file(self):
        return h5py.File(self._data_file_name, 'r')

    @property
    def _tmp_file(self):
        return h5py.File(os.path.join(self._fopts_tmp_dir, "tmp.h5"))

    def _iter_seeds(self, ds):
        assert ds in ['tr','va','ts']
        if not hasattr(self, '_seeds'):
            raise RuntimeError("Can't iter seeds without seeds initialized.")
        for key in self._seeds[ds]:
            for iseed,seed in enumerate(self._seeds[ds][key]):
                yield key,iseed,seed

    def _iter_tmp(self):
        for ds in ['tr','va','ts']:
            for key,iseed,seed in self._iter_seeds(ds):
                yield ds, key, iseed, seed

    def _initialize(self):
        df = self._data_file
        tf = self._tmp_file

        step = np.inf

        nexamples = len(list(self._iter_tmp()))

        for index,(ds,key,iseed,seed) in enumerate(self._iter_tmp()):
            self._logger.progress("... initializing", index, nexamples)

            # Create dataset group if it doesn't exist.
            if ds not in tf:
                tf.create_group(ds)

            img = df[key+'/img'][...]

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            # Compute the initialization for this example and seed value.
            u0, dist, mask = self.init_func(img, self.band,
                                            dx=df[key].attrs['dx'],
                                            seed=seed)

            # "Auto" step: only use training and validation datasets.
            if ds in ['tr', 'va']:
                # Compute the maximum velocity for the i'th example
                mx = np.abs(df[key+"/dist"][mask]).max()

                # Create the candidate step size for this example.
                tmp = np.min(df[key].attrs['dx']) / mx

                # Assign tmp to step if it is the smallest observed so far.
                step = tmp if tmp < step else step

            # Create a group for the i'th example.
            if key not in tf[ds]:
                tf[ds].create_group(key)

            seed_group = tf[ds][key].create_group("seed-%d" % iseed)

            # The group consists of the current "level set field" u, the 
            # signed distance transform of u (only in the narrow band), and
            # the boolean mask indicating the narrow band region.
            seed_group.create_dataset("u",    data=u0,   compression='gzip')
            seed_group.create_dataset("dist", data=dist, compression='gzip')
            seed_group.create_dataset("mask", data=mask, compression='gzip')

            if self._fopts_model_fit_method == 'rf':
                nu = np.zeros_like(dist)
                seed_group.create_dataset("nu", data=nu,
                                          compression='gzip')

        if self.step == 'auto':
            self.step = step
            self._logger.info("Computed auto step is %.7f." % self.step)
        elif self.step != 'auto' and self.step > step:
            self._logger.warn("Computed step is %.7f but given step is %.7f."
                                % (step, self.step))
        df.close()
        tf.close()

    def _update_level_sets(self):
        df = self._data_file
        tf = self._tmp_file

        if self._fopts_model_fit_method == 'nnet':
            nnet = self.models[-1]

        # Loop over all indices in the validation dataset.
        for ds,key,iseed,seed in self._iter_tmp():
            img = df[key+"/img"][...]

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            dx = df[key].attrs['dx']

            u    = tf["%s/%s/seed-%d/u"    % (ds, key, iseed)][...]
            dist = tf["%s/%s/seed-%d/dist" % (ds, key, iseed)][...]
            mask = tf["%s/%s/seed-%d/mask" % (ds, key, iseed)][...]

            # Don't update if the mask is empty.
            if not mask.any():
                continue

            if self._fopts_model_fit_method != 'rf':
                # Compute features.
                F = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

                # Compute approximate velocity from features.
                nu = np.zeros_like(u)
                nu[mask] = nnet.predict(F[mask])
            else:
                nu = tf["%s/%s/seed-%d/nu" % (ds, key, iseed)][...]

            # Compute gradient magnitude using upwind Osher/Sethian method.
            gmag = mg.gmag_os(u, nu, mask=mask, dx=dx)

            # Here's the actual level set update.
            u[mask] += self.step*nu[mask]*gmag[mask]

            # Update the distance transform and mask,
            # checking if the zero level set has vanished.
            if (u > 0).all() or (u < 0).all():
                mask = np.zeros_like(mask)
                dist = np.zeros_like(dist)
            else:
                # Update the distance and mask for u.
                dist = skfmm.distance(u, narrow=self.band, dx=dx)
                
                if hasattr(dist, 'mask'):
                    mask = ~dist.mask
                    dist = dist.data
                else:
                    # This might happend if band is very large
                    # or the object is very large.
                    mask = np.ones(dist.shape, dtype=np.bool)

            # Update the data in the hdf5 file.
            tf["%s/%s/seed-%d/u"    % (ds, key, iseed)][...] = u
            tf["%s/%s/seed-%d/dist" % (ds, key, iseed)][...] = dist
            tf["%s/%s/seed-%d/mask" % (ds, key, iseed)][...] = mask

        df.close()
        tf.close()

    def _validation_dummy_update(self, nnet):
        """
        Do a "dummy" update over the validation dataset to 
        record segmentation scores.
        """
        df = self._data_file
        tf = self._tmp_file

        mu = 0.

        # Loop over all indices in the validation dataset.
        for key,iseed,seed in self._iter_seeds('va'):
            img = df[key+"/img"][...]

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            seg = df[key+"/seg"][...]
            dx  = df[key].attrs['dx']

            u    = tf["va/%s/seed-%d/u"    % (key,iseed)][...]
            dist = tf["va/%s/seed-%d/dist" % (key,iseed)][...]
            mask = tf["va/%s/seed-%d/mask" % (key,iseed)][...]

            if mask.any():
                if self._fopts_model_fit_method != 'rf':
                    # Compute features.
                    F = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

                    # Compute approximate velocity from features.
                    nu = np.zeros_like(u)
                    nu[mask] = nnet.predict(F[mask])
                else:
                    nu = tf["%s/%s/seed-%d/nu" % (ds, key, iseed)][...]

                # Compute gradient magnitude using 
                # upwind Osher/Sethian method.
                gmag = mg.gmag_os(u, nu, mask=mask, dx=dx)

                # Here's the dummy update.
                utmp = u.copy()
                utmp[mask] += self.step*nu[mask]*gmag[mask]

            mu += self.score_func(utmp, seg)*1.0 / self._nva

        df.close()
        tf.close()

        return mu

    def _balance_mask(self, y, rs=None):
        """
        Return a mask for balancing `y` to have equal negative and positive.
        """
        rs = np.random.RandomState() if rs is None else rs

        n = y.shape[0]
        npos = (y > 0).sum()
        nneg = (y < 0).sum()

        if npos > nneg: # then down-sample the positive elements
            wpos = np.where(y > 0)[0]
            inds = rs.choice(wpos, replace=False, size=nneg)
            mask = y <= 0
            mask[inds] = True
        elif npos < nneg: # then down-sample the negative elements.
            wneg = np.where(y < 0)[0]
            inds = rs.choice(wneg, replace=False, size=npos)
            mask = y >= 0
            mask[inds] = True
        else: # n = npos or n == nneg or npos == nneg
            mask = np.ones(y.shape, dtype=np.bool)

        return mask

    def _featurize_all_images(self, dataset, balance, rs):
        """
        dataset: str
            Should be in ['tr','va','ts']

        balance: bool
            Balance by neg/pos of target value.

        rs: numpy.RandomState
            To make results reproducible. The random state
            should be passed here rather than using the `self._rs`
            attribute, since in the multiprocessing setting we can't
            rely on `self._rs`.
        """
        assert dataset in ['tr','va','ts']
        who = ['tr','va','ts'].index(dataset)

        tf = self._tmp_file

        # Count the number of points collected
        count = 0

        for ds,key,iseed,seed in self._iter_tmp():
            if ds != dataset: continue

            with self._data_file as df:
                img    = df[key+"/img"][...]
                target = df[key+"/dist"][...]
                dx     = df[key].attrs['dx']

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            u    = tf["%s/%s/seed-%d/u"    % (dataset, key, iseed)][...]
            dist = tf["%s/%s/seed-%d/dist" % (dataset, key, iseed)][...]
            mask = tf["%s/%s/seed-%d/mask" % (dataset, key, iseed)][...]

            if not mask.any():
                continue # Otherwise, repeat the loop until a non-empty mask.

            # Precompute data size.
            if balance:
                npos = (target[mask] > 0).sum()
                nneg = (target[mask] < 0).sum()
                if min(npos,nneg) > 0:
                    count += 2*min(npos,nneg)
                else:
                    count += max(npos, nneg)
            else:
                count += mask.sum()

        X = np.zeros((count, self.feature_map.nfeatures))
        y = np.zeros((count,))

        index = 0

        for ds,key,iseed,seed in self._iter_tmp():
            if ds != dataset: continue

            with self._data_file as df:
                img    = df[key+"/img"][...]
                target = df[key+"/dist"][...]
                dx     = df[key].attrs['dx']

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            u    = tf["%s/%s/seed-%d/u"    % (dataset, key, iseed)][...]
            dist = tf["%s/%s/seed-%d/dist" % (dataset, key, iseed)][...]
            mask = tf["%s/%s/seed-%d/mask" % (dataset, key, iseed)][...]

            if not mask.any():
                continue # Otherwise, repeat the loop until a non-empty mask.

            # Compute features.
            features = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

            if balance:
                bmask = self._balance_mask(target[mask], rs=rs)
                next_index = index + target[mask][bmask].shape[0]
                X[index:next_index] = features[mask][bmask]
                y[index:next_index] =   target[mask][bmask]
            else:
                next_index = index + mask.sum()
                X[index:next_index] = features[mask]
                y[index:next_index] =   target[mask]

            index = next_index

        tf.close()

        return X, y

    def _featurize_random_image(self, dataset, balance, rs):
        """
        dataset: str
            Should be in ['tr','va','ts']

        balance: bool
            Balance by neg/pos of target value.

        rs: numpy.RandomState
            To make results reproducible. The random state
            should be passed here rather than using the `self._rs`
            attribute, since in the multiprocessing setting we can't
            rely on `self._rs`.
        """
        assert dataset in ['tr','va','ts']
        who = ['tr','va','ts'].index(dataset)

        while True:
            # Get a random image from the appropriate dataset.
            key   = rs.choice(self._seeds[dataset].keys())
            iseed = rs.choice(len(self._seeds[dataset][key]))

            with self._data_file as df:
                img    = df[key+"/img"][...]
                target = df[key+"/dist"][...]
                dx     = df[key].attrs['dx']

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            with self._tmp_file as tf:
                u    = tf["%s/%s/seed-%d/u"    % (dataset, key, iseed)][...]
                dist = tf["%s/%s/seed-%d/dist" % (dataset, key, iseed)][...]
                mask = tf["%s/%s/seed-%d/mask" % (dataset, key, iseed)][...]

            if mask.any():
                break # Otherwise, repeat the loop until a non-empty mask.

        # Precompute data size.
        if balance:
            npos = (target[mask] > 0).sum()
            nneg = (target[mask] < 0).sum()
            if min(npos,nneg) > 0:
                count = 2*min(npos,nneg)
            else:
                count = max(npos, nneg)
        else:
            count = mask.sum()

        X = np.zeros((count, self.feature_map.nfeatures))
        y = np.zeros((count,))

        # Compute features.
        features = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

        if balance:
            bmask = self._balance_mask(target[mask], rs=rs)
            X = features[mask][bmask]
            y =   target[mask][bmask]
        else:
            X = features[mask]
            y =   target[mask]

        # Shuffle the data.
        state = rs.get_state()
        rs.shuffle(X)
        rs.set_state(state)
        rs.shuffle(y)

        return X, y

    def _fit_nnet(self, nh, init):
        """
        Fit a neural network model.
        """
        # Create a unique random state seed from the inputs
        # since we shouldn't rely on `self._rs` in the multiprocess setting.
        rs_seed = (nh+init+1)*(nh+init)/2 + init # <= Cantor pairing function
        rs = np.random.RandomState(rs_seed)

        nnet = nn.neural_network(self.feature_map.nfeatures, nh, rs=rs)
        nnet.loss_tr = np.zeros(self._nopts_maxepochs+1)
        nnet.loss_va = np.zeros(self._nopts_maxepochs+1)

        nnet_name = "nnet-nh=%d-init=%d.pkl" % (nh, init)
        gradstep  = self._nopts_step
        iterspb   = self._nopts_iters_per_batch
        balance   = self._nopts_balance

        logger = fit_logger(file=os.path.join(self._iter_dir, 
                                              nnet_name[:-3]+"log"),
                            stdout=False, stamp=False)

        if self._nopts_batch_size_auto:
            bs = self._auto_batch_size
        else:
            bs = self._nopts_batch_size

        if self._nopts_batches_per_epoch is None:
            bpe = self._auto_batches_per_epoch
        else:
            bpe = self._nopts_batches_per_epoch

        bqtr = nn.batch_queue(bs)
        bqva = nn.batch_queue(bs)

        lstr = "Epoch %d / %d, Batch %d / %d."

        # Stochastic gradient descent.
        for epoch in range(self._nopts_maxepochs+1):
            for batch in range(bpe):
                if self._nopts_log_batch:
                    logger.info(lstr % (epoch,
                                        self._nopts_maxepochs,
                                        batch+1,
                                        self._nopts_batches_per_epoch))

                while len(bqtr) == 0:
                    x,y = self._featurize_random_image(dataset='tr', 
                                                       balance=balance, rs=rs)
                    bqtr.update(x,y)

                while len(bqva) == 0:
                    x,y = self._featurize_random_image(dataset='va',
                                                       balance=balance, rs=rs)
                    bqva.update(x,y)

                Xtr,ytr = bqtr.next()
                Xva,yva = bqva.next()

                # The initial epoch only computes loss.
                if epoch == 0:
                    ltr = nnet.loss(Xtr,ytr)
                    lva = nnet.loss(Xva,yva)
                else:
                    ltr,lva = nnet.gradient_descent(Xtr, ytr, Xva, yva,
                                                    step=gradstep, 
                                                    iters=iterspb,
                                                    ret_last=True,
                                                    verbose=False,
                                                    logger=logger)
                nnet.loss_tr[epoch] += ltr / bpe
                nnet.loss_va[epoch] += lva / bpe

            logger.progress("Epoch. LTR=%.4f, LVA=%.4f" 
                             % (nnet.loss_tr[epoch], nnet.loss_va[epoch]),
                             epoch, self._nopts_maxepochs)

            # Save the model if the validation loss is the best observed.
            if epoch == 0 or nnet.loss_va[epoch] < nnet.loss_va[:epoch].min():
                with open(os.path.join(self._iter_dir, nnet_name), 'w') as f:
                    pickle.dump(nnet, f)

            # Check for early stop conditions.
            if nn.stop_early(nnet.loss_va[:epoch],
                             self._nopts_va_hist_len,
                             self._nopts_va_hist_tol):
                loss_tr = nnet.loss_tr.copy()[:epoch+1]
                loss_va = nnet.loss_va.copy()[:epoch+1]

                # Load the model that had the best parameters under MSE on va.
                with open(os.path.join(self._iter_dir, nnet_name)) as f:
                    nnet = pickle.load(f)

                # Attach the latest loss records to the best model and save.
                nnet.loss_tr = loss_tr
                nnet.loss_va = loss_va

                with open(os.path.join(self._iter_dir, nnet_name), 'w') as f:
                    pickle.dump(nnet, f)

                # Break the epoch loop.
                break

        # Compute the average segmentation score over validation and save it.
        avg_seg_score_va = self._validation_dummy_update(nnet)
        np.save(os.path.join(self._iter_dir, nnet_name[:-3]+"npy"),
                avg_seg_score_va)

        logger.handlers[0].close()
        del logger

    def _compute_auto_batch_size(self):
        df = self._data_file
        tf = self._tmp_file

        mu = 0.0
        balance = self._nopts_balance

        for key,iseed,seed in self._iter_seeds('tr'):
            mask = tf["tr/%s/seed-%d/mask" % (key,iseed)][...]
            targ = df[key+"/dist"][...]

            if not balance:
                mu += mask.sum()*1.0 / self._ntr
            else:
                npos = (targ[mask] > 0).sum()
                nneg = (targ[mask] < 0).sum()
                
                if 0 < nneg < npos:
                    mu += 2.0*nneg / self._ntr
                elif 0 < npos < nneg:
                    mu += 2.0*npos / self._ntr
                else:
                    mu += max(npos, nneg) / self._ntr

        self._auto_batch_size = int(mu) * self._nopts_batch_size
        self._logger.info("Computed auto batch size is %d."
                                 % self._auto_batch_size)

        df.close()
        tf.close()

    def _compute_auto_batches_per_epoch(self):
        df = self._data_file
        tf = self._tmp_file

        total = 0.0
        balance = self._nopts_balance

        for key,iseed,seed in self._iter_seeds('tr'):
            mask = tf["tr/%s/seed-%d/mask" % (key,iseed)][...]
            targ = df[key+"/dist"][...]

            if not balance:
                total += mask.sum()
            else:
                npos = (targ > 0).sum()
                nneg = (targ < 0).sum()
                
                if 0 < nneg < npos:
                    total += 2*nneg
                elif 0 < npos < nneg:
                    total += 2*npos
                else:
                    total += max(npos, nneg)

        if self._nopts_batch_size_auto:
            self._auto_batches_per_epoch = int(total / self._auto_batch_size)
        else:
            self._auto_batches_per_epoch = int(total / self._nopts_batch_size)

        # Make sure it's at least 1 batch per epoch.
        self._auto_batches_per_epoch = max(1, self._auto_batches_per_epoch)

        self._logger.info("Computed auto batches per epoch is %d."
                                 % self._auto_batches_per_epoch)

        df.close()
        tf.close()
        
    def _fit_model(self):
        """
        Fit a regression model for advancing the level set.
        """
        self._iter_dir = os.path.join(self._models_path, 
                                      "iter-%d" % self._iter)
        os.mkdir(self._iter_dir)

        if self._fopts_model_fit_method == 'nnet':
            if self._nopts_batch_size_auto:
                self._compute_auto_batch_size()

            if self._nopts_batches_per_epoch is None:
                self._compute_auto_batches_per_epoch()

            # We'll start a process for each trial of 
            # hidden unit number and init.
            procs = []
            self._logger.info("Training neural nets. Logs are in %s."
                                    % self._iter_dir)
            for nh in self._nopts_nhidden:
                for init in range(self._nopts_ninits):
                    p = mp.Process(target=self._fit_nnet,
                                   args=(nh, init),
                                   name="nnet: nh=%d, init=%d" % (nh, init))
                    p.start()
                    procs.append(p)

            for p in procs:
                p.join()
                if p.exitcode != 0:
                    msg = ("An error occured during nnet training (%s)." 
                            % p.name)
                    self._logger.error(msg)
                    for q in procs: q.terminate()
                    raise RuntimeError(msg)

            bestnh    = None
            bestinit  = None
            bestscore = -np.inf

            # Determine best model over validation data scores.
            for inh,nh in enumerate(self._nopts_nhidden):
                for init in range(self._nopts_ninits):
                    score = np.load(os.path.join(self._iter_dir,
                                                 "nnet-nh=%d-init=%d.npy"
                                                 % (nh, init)))
                    score = np.asscalar(score) # because we saved as numpy array.
                    if score > bestscore:
                        bestnh    = nh
                        bestinit  = init
                        bestscore = score

            self._logger.info("Nnet train done. Best nhidden is %d." % bestnh)

            # Load the best model and add it to the model list.
            nnet = np.load(os.path.join(self._iter_dir,
                                        "nnet-nh=%d-init=%d.pkl"
                                        % (bestnh, bestinit)))

            if hasattr(self, 'models'):
                self.models.append(nnet)
            else:
                self.models = [nnet]

        elif self._fopts_model_fit_method == 'rf':

            self._set_all_nu_zero()

            self._logger.info('Featurizing training images.')

            # Get input and output variables
            X, y = self._featurize_all_images('tr',
                                              self._rfopts_balance,
                                              self._rs)

            np.save(os.path.join(self._fopts_tmp_dir, 'X.npy'), X)
            np.save(os.path.join(self._fopts_tmp_dir, 'y.npy'), y)

            for itree in range(self._rfopts_n_estimators):

                self._logger.progress("Fitting tree.", itree+1,
                                      self._rfopts_n_estimators)

                # Fit in a new process for memory release
                p = mp.Process(target=self._fit_tree,
                               args=(itree,),
                               name="tree fit (%d)" % (itree+1))
                p.start()
                p.join()

                if p.exitcode != 0:
                    msg = ("An error occured during tree training (%s)." 
                            % p.name)
                    self._logger.error(msg)
                    raise RuntimeError(msg)

            # Append the computed feature importances
            fimp = np.load(os.path.join(self._fopts_tmp_dir, 'fimp.npy'))
            if not hasattr(self, 'feature_importances'):
                self.feature_importances = [fimp]
            else:
                self.feature_importances.append(fimp)

        else:
            raise NotImplementedError

        # Remove the temporary neural net files if necessary.
        if self._fopts_remove_tmp:
            self._logger.info("Removing tmp files at %s." % self._iter_dir)
            shutil.rmtree(self._iter_dir)
        del self._iter_dir

    def _fit_tree(self, itree):

        X = np.load(os.path.join(self._fopts_tmp_dir, 'X.npy'))
        y = np.load(os.path.join(self._fopts_tmp_dir, 'y.npy'))

        rs = np.random.RandomState(itree)

        dtr = DecisionTreeRegressor(
            max_features=self._rfopts_max_features,
            max_depth=self._rfopts_max_depth,
            criterion=self._rfopts_criterion,
            random_state=rs,
        )

        # Fit the decision tree regression model
        dtr.fit(X, y)

        # Update nu
        self._update_nu(dtr)

        if itree == 0:
            np.save(os.path.join(self._fopts_tmp_dir, 'fimp.npy'),
                    dtr.feature_importances_)
        else:
            fimp = np.load(os.path.join(self._fopts_tmp_dir, 'fimp.npy'))
            fimp += dtr.feature_importances_ / self._rfopts_n_estimators
            np.save(os.path.join(self._fopts_tmp_dir, 'fimp.npy'), fimp)

    def _update_nu(self, dtr):
        df = self._data_file
        tf = self._tmp_file
        n_estimators = self._rfopts_n_estimators

        for ds,key,iseed,seed in self._iter_tmp():
            dx = df[key].attrs['dx']
            img = df[key + '/img'][...]

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            u    = tf[ds][key]['seed-%d'%iseed]['u'   ][...]
            mask = tf[ds][key]['seed-%d'%iseed]['mask'][...]
            dist = tf[ds][key]['seed-%d'%iseed]['dist'][...]

            if not mask.any():
                continue

            features = self.feature_map(u=u, img=img, 
                                        mask=mask, dist=dist,
                                        dx=dx)

            tf[ds][key]['seed-%d'%iseed]['nu'][mask] += \
                    dtr.predict(features[mask]) /n_estimators


        tf.close()
        df.close()

    def _set_all_nu_zero(self):
        tf = self._tmp_file

        for ds,key,iseed,seed in self._iter_tmp():
            tf[ds][key]['seed-%d'%iseed]['nu'][...] = 0.

        tf.close()

    def _check_early_exit(self):
        i = self._iter
        l = self._fopts_va_hist_len
        t = self._fopts_va_hist_tol

        if i >= l-1:
            if not hasattr(self, '_va_hist_x'):
                self._va_hist_x = np.c_[np.ones(l), np.arange(l)]
            x = self._va_hist_x
            s = self._scores_over_iters('va')[0].mean(0)

            # Scores over past `l` iters (current iteration inclusive).
            y = s[i+1-l:i+1]

            # The slope of the best fit line.
            m = np.linalg.lstsq(x, y, rcond=None)[0][1]

            # True if slope is less than tolerance.
            return m < t, m
        else:
            return False, None

    def fit(self):
        """
        Run `set_fit_options` and `set_net_opts` before calling `fit`.
        """
        if not self._fit_opts_set:
            raise RuntimeError("Run `set_fit_options` before fitting.")
        if self._fopts_model_fit_method == 'nnet' and not self._net_opts_set:
            raise RuntimeError("Run `set_net_opts` before fitting.")
        if self._fopts_model_fit_method == 'rf' and not self._rf_opts_set:
            raise RuntimeError("Run `set_rf_opts` before fitting.")
        if self._is_fitted:
            raise RuntimeError("This model has already been fitted.")
        

        self._logger.info("Initializing.")
        self._initialize()
        self._iter = 0

        self._logger.info("Collecting scores.")
        self._collect_scores()
        self._logger.progress(
            "Scores => TR: %.4f VA: %.4f TS: %.4f."
                % tuple(self._get_mean_scores_at_iter(self._iter)),
            self._iter,
            self._fopts_maxiters
        )

        self._models_path = os.path.join(self._fopts_tmp_dir, "models")
        os.mkdir(self._models_path)

        maxiters = self._fopts_maxiters
        start_iter = 1


        for self._iter in range(start_iter, maxiters+1):
            self._logger.progress("Beginning iteration.",
                                  self._iter, maxiters)
            iter_start = time.time()

            self._logger.info("Fitting model with method %s."
                                % self._fopts_model_fit_method)
            self._fit_model()

            self._logger.progress("Updating level sets.",
                                  self._iter, maxiters)
            self._update_level_sets()

            self._logger.progress("Collecting scores.", self._iter, maxiters)
            self._collect_scores()
            self._logger.progress("Scores => TR: %.4f VA: %.4f TS: %.4f."
                          % tuple(self._get_mean_scores_at_iter(self._iter)),
                                  self._iter, maxiters)

            iter_stop = time.time()
            iter_time = (iter_stop-iter_start) / 3600.

            self._logger.progress(("End of iteration. Ellapsed time: "
                                   "%.3f hours.") % iter_time,
                                  self._iter, maxiters)

            self._logger.info("Saving model to %s." % self._fopts_save_file)

            # Swap the logger before saving since file objects don't pickle.
            logger = self._logger
            self._logger = None

            # Save the model. 
            with open(self._fopts_save_file, 'w') as f:
                pickle.dump(self, f)

            # Swap the logger back.
            self._logger = logger

            exit_early, trend = self._check_early_exit()
            if trend is not None:
                self._logger.info("Linear trend over VA: %.4f (tol=%.4f)."
                                    % (trend, self._fopts_va_hist_tol))
                if exit_early:
                    self._logger.info("Trend less than tolerance. Exiting.")
                    break

        if self._fopts_remove_tmp:
            self._logger.info("Removing temporary files at %s." 
                                    % self._fopts_tmp_dir)
            shutil.rmtree(self._fopts_tmp_dir)

        scores = self._scores_over_iters('va')[0].mean(0)
        nmax = scores.argmax()

        if self._fopts_model_fit_method != 'rf':
            self.models = self.models[:nmax] if nmax > 0 else []

        self._is_fitted = True
        self._logger.info("Fitting complete.")

        self._logger.info("Saving model to %s." % self._fopts_save_file)

        # Close I/O and file stream handlers before deleting the logger.
        for h in self._logger.handlers: h.close()
        del self._logger

        # Save the model. 
        with open(self._fopts_save_file, 'w') as f:
            pickle.dump(self, f)


    def set_fit_options(self, datasets=None, seeds=None, save_file=None,
                        model_fit_method="nnet",
                        normalize_images=True, tmp_dir=None, remove_tmp=False,
                        maxiters=100, va_hist_len=5, va_hist_tol=0.0,
                        logfile=None, logstamp=True, logstdout=True):
        """
        Parameters
        ----------
        datasets: dict, len=3, default=None
            This should be a dict of 3 lists of the form::

                datasets['tr'] = list of strings
                datasets['va'] = list of strings
                datasets['ts'] = list of strings

            Each list of strings contains the keys into the h5py dataset
            file for training ('tr'), validation ('va'), and testing ('ts').
            So for example, `datasets['tr'][0]` yields a string that is a 
            key into the dataset file for the first training example.

            If `datasets` is None (default), then an auto split of 
            60/20/20% is made using the 
            `slls.utils.data.splitter` utility.

        seeds: dict, default=None
            The seeds are used in the `init_func` supplied at initialization.
            Note that seed coordinates must be given in *index* coordinates, 
            i.e., they should *not* take into account pixel spacing / 
            resolution. However, fractional values are allowed (float type).

            `seeds` format:

            Let `ds` be in `['tr','va','ts']`. For each `key` (string) in 
            `datasets[ds]`, we should have `seeds[ds][key]` being an iterable 
            of iterables. `seeds[ds][key]` refers to the `key` example in 
            dataset `ds`, and thus `seeds[ds][key]` should contain a list 
            of seeds for that example (each seed being an iterable type).
            
            For example, we might have `seeds[ds][key] == [[25,34], [27,35]]`, 
            which indicates there are two seeds for example `key` in 
            dataset `ds`.

            If None (default), then a single seed for each example is used 
            and computed automatically: the center-of-mass of the ground-
            truth segmentation of the respective examples.

        save_file: str, default=None
            The model will be pickled to this path.
            The default (None) uses the file `slls_model.pkl` at the current
            working directory.

        model_fit_method: str, default='nnet'
            'nnet': uses neural network regression with online stocastic
                    gradient descent for optimization.
            'rf': uses random forest regression. NOTE: in this
                  case, the regression models are *not* saved since tree
                  models tend to be quite large in terms of data storage.

        normalize_images: bool, default=True
            If True, then whenever an image is loaded, we normalize it, i.e.,
            we compute and use `(img - img.mean()) / img.std()` in place of
            each image array, `img`.

        tmp_dir: str, default=None
            The working directory. Temporary data (e.g., intermediate
            level set values) will be stored in this directory.
            The default uses the current working directory.

        maxiters: int, default=100
            The maximum number of iterations (i.e., number of 
            regression models).  Note that less than `maxiters` will be used 
            if the validation set performance degrades (see `va_his_len` 
            and `va_hist_tol`).

        remove_tmp: bool, default=False
            If True, the temporary files created over the fitting process 
            are left in place and not deleted. This includes for example,
            all the network models for the grid search procedure.

        va_hist_len: int, default=5
            The process terminates when the average segmentation quality 
            over the last `va_hist_len` iterations is trending downward, or
            when `maxiters` has been reached.

        va_hist_tol: float, default=0.0
            The linear trend over of the average segmentation quality over 
            the last `va_hist_len` iterations is computed. The process 
            terminates if the linear trend (slope of best fit line) is less 
            than `va_hist_tol`, or when `maxiters` has been reached.

        logfile: str, default=None
            The name of the log file to write. The default (None) writes to
            `log.txt` in the current working directory.

        logstamp: bool, default=True
            Include a timestamp in the `logfile` name.

        logstdout: bool, default=True
            If True, the log will print to stdout (as well as logging to file).
        """
        if model_fit_method not in MODEL_FIT_METHODS:
            msg = 'model_fit_method ({}) must be one of {}.'
            raise ValueError(msg.format(model_fit_method, MODEL_FIT_METHODS))

        # This is hackish. It just sets most the arguments to member variables.
        L = locals(); L.pop('self'); L.pop('datasets'); L.pop('seeds')
        for l in L: setattr(self, "_fopts_%s"%l, L[l])

        self._validate_datasets(datasets)
        self._validate_seeds(seeds)

        # Count the number of examples.
        self._ntr = len([_ for _ in self._iter_seeds('tr')])
        self._nva = len([_ for _ in self._iter_seeds('va')])
        self._nts = len([_ for _ in self._iter_seeds('ts')])
        self._ntotal = self._ntr + self._nva + self._nts


        if self._fopts_save_file is None:
            self._fopts_save_file = os.path.join(os.path.curdir,
                                                 "slls_model.pkl")

        self._fopts_save_file = os.path.abspath(
            self._fopts_save_file
        )

        if self._fopts_tmp_dir is None:
            self._fopts_tmp_dir = os.path.join(os.path.curdir, 'tmp')
        else:
            self._fopts_tmp_dir = os.path.join(self._fopts_tmp_dir, 'tmp')

        self._fopts_tmp_dir = os.path.abspath(self._fopts_tmp_dir)

        os.mkdir(self._fopts_tmp_dir)

        self._logger = fit_logger(file=logfile, stamp=logstamp, 
                                  stdout=logstdout)
        self._fit_opts_set = True

    def set_net_options(self, nhidden=[64, 128, 256, 512], ninits=3,
                        step=0.1, maxepochs=1000, batch_size=5,
                        batch_size_auto=True,
                        batches_per_epoch=None, iters_per_batch=1, 
                        log_batch=False, balance=True, 
                        va_hist_len=25, va_hist_tol=0.0):
        """
        nhidden: list of ints, default=[64, 128, 256, 512]
            The number hidden hidden units to search over.

        ninits: int, default=3
            The number of random initializations to perform for optimization 
            using each `nhidden` value.

        step: float, default=0.1
            The step size for stochastic gradient descent.

        maxepochs: int, default=1000
            The maximum allowable number of epochs.

        batch_size: int, default=5
            The batch size for stochastic gradient descent.
            If `batch_size_image` is True (default), then `batch_size` 
            should be interpreted as the (average) number of images 
            in a single batch (i.e., the average number number of images 
            used to collect feature vectors to compose a single batch).
            Otherwise (`batch_size_image` False), `batch_size`
            is interpretted as the number of feature vector examples 
            to be included in a single batch.

        batch_size_image: bool, default=True
            See `batch_size`.

        batches_per_epoch: int, default=None
            The number of batch updates performed in each "epoch".
            If None (default), then the effective number of batches
            necessary to make one pass through the training data
            (on average) is computed and used.

        iters_per_batch: int, default=1
            The number of gradient descent updates to perform with each batch.

        log_batch: bool, default=False
            If True, then the log file for each neural network training
            session records the current batch number. This is in addition 
            to recording the loss values over each epoch. Setting this 
            flag to True is useful if you expect batches to take a long time, 
            which in general should not be the case.

        balance: bool, default=True
            If True, the each batch is balanced by the positivity/negativity
            of the target values in the batch.

        va_hist_len: int, default=5
            The process terminates when the mean-squared error over the 
            validation dataset over the last `va_hist_len` epochs is 
            trending downward, or when `maxepochs` has been reached.

        va_hist_tol: float, default=0.0
            The linear trend over of the MSE over 
            the last `va_hist_len` epochs is computed. The process terminates 
            if the linear trend (slope of best fit line) is less than 
            `va_hist_tol`, or when `maxepochs` has been reached.
        """
        # This is hackish. It just sets all the arguments to member variables.
        L = locals(); L.pop('self')
        for l in L: setattr(self, "_nopts_%s"%l, L[l])
        self._net_opts_set = True
        if batch_size_auto and batch_size > 50:
            warnings.warn(("`batch_size_auto` is True, but `batch_size` "
                           "> 50. This could result in excessive train "
                           "times and/or memory consumption."))

    def set_rf_opts(self, n_estimators=100, max_features="auto",
                    max_depth=None, criterion="mse",
                    balance=True):
        L = locals(); L.pop('self')
        for l in L: setattr(self, "_rfopts_%s"%l, L[l])
        self._rf_opts_set = True

    def segment(self, img, seg=None, dx=None, verbose=True, memoize=True):
        """
        Segment `img`.

        Parameters
        ----------
        img: ndarray
            The image.

        seg: ndarry, default=None
            The boolean segmentation volume. If provided (not None), then
            the score for the model against the ground-truth `seg` is 
            computed and returned.

        dx: ndarray or list, default=None
            List of the spatial delta terms along each axis. The default 
            uses ones.

        verbose: bool, default=True
            Print progress.

        Returns
        -------
        u[, scores]: ndarray[, ndarray]
            `u` is shape `(len(self.models)+1,) + img.shape`, where `u[i]`
            is the i'th iterate of the level set function and `u[i] > 0` 
            yields an approximate boolean-mask segmentation of `img` at
            the i'th iteration.
        """
        if not self._is_fitted:
            raise RuntimeError("This model hasn't been fit yet.")
        if self._fopts_model_fit_method == 'rf':
            raise RuntimeError("Model are not stored for `rf` fit method.")

        # If normalize, assign to `img_` so as to not overwrite `img`.
        if self._fopts_normalize_images:
            img_ = (img - img.mean()) / img.std()
        else:
            img_ = img

        iters = len(self.models)
        dx = np.ones(img_.ndim) if dx is None else dx

        if dx.shape[0] != img_.ndim:
            raise ValueError("`dx` has incorrect number of elements.")

        u = np.zeros((iters+1,) + img_.shape)
        u[0], dist, mask = self.init_func(img_, self.band, dx=dx)

        if seg is not None:
            scores = np.zeros((iters+1,))
            scores[0] = self.score_func(u[0], seg)

        nu = np.zeros(img_.shape)

        if verbose:
            if seg is None:
                pstr = "Iter: %02d"
                print(pstr % 0)
            else:
                pstr = "Iter: %02d, Score: %0.5f"
                print(pstr % (0, scores[0]))

        for i in range(iters):
            u[i+1] = u[i].copy()

            if mask.any():
                # Compute the features, and use the model to predict speeds.
                mem = 'create' if i == 0 else 'use'
                mem = mem if memoize else None

                features = self.feature_map(u[i], img_, dist=dist,
                                            mask=mask, memoize=mem, dx=dx)

                nu[mask] = self.models[i].predict(features[mask])

                gmag = mg.gmag_os(u[i], nu, mask=mask, dx=dx)
                    
                # Update the level set.
                u[i+1][mask] += self.step*nu[mask]*gmag[mask]

                # Check for level set vanishing.
                if (u[i+1] > 0).all() or (u[i+1] < 0).all():
                    mask = np.zeros_like(mask)
                    dist = np.zeros_like(dist)
                else:
                    # Update the signed distance function for phi.
                    dist = skfmm.distance(u[i+1], narrow=self.band, dx=dx)

                    if hasattr(dist, 'mask'):
                        mask = ~dist.mask
                        dist =  dist.data
                    else:
                        mask = np.ones(u[i+1].shape, dtype=np.bool)

            if seg is not None:
                scores[i+1] = self.score_func(u[i+1], seg)

            if verbose and seg is None:
                print(pstr % (i+1))
            elif verbose and seg is not None:
                print(pstr % (i+1, scores[i+1]))

        if seg is None:
            return u
        else:
            return u, scores
    
    def _validate_datasets(self, datasets):
        if datasets is None:
            df = self._data_file
            datasets = splitter.split(df.keys(), rs=self._rs)
            df.close()

        # Check if all dataset keys are present.
        for d in ['tr', 'va', 'ts']:
            if d not in datasets:
                raise ValueError("Missing %s dataset."%d)

        if not all([np.iterable(datasets[d]) for d in datasets]):
            raise ValueError("Each dataset should be an iterable of keys.")

        if len(set(datasets['tr']).intersection(set(datasets['va']))) > 0:
            raise ValueError("Training set keys overlap with validation.")

        if len(set(datasets['tr']).intersection(set(datasets['ts']))) > 0:
            raise ValueError("Training set indices overlap with testing.")

        if len(set(datasets['va']).intersection(set(datasets['ts']))) > 0:
            raise ValueError("Validation set indices overlap with testing.")

        self._datasets = datasets

    def _validate_seeds(self, seeds):
        if seeds is None:
            # Default behavior: use compute and use a single seed
            # (the ground truth center of mass) for each.

            df = self._data_file
            seeds = dict(tr={}, va={}, ts={})

            for ds in ['tr','va','ts']:
                for key in self._datasets[ds]:
                    seg = df[key+"/seg"][...]

                    # Get the index coordinates.
                    inds = np.indices(seg.shape, dtype=np.float)

                    total = seg.sum()

                    # Compute the center of mass as the seed.
                    seed = [(i*seg).sum() / total for i in inds]

                    # Append the single seed as a list of length 1,
                    # since, in general, the seed list can be longer for
                    # a single example (i.e., for a single index).
                    seeds[ds][key] = [seed]
            df.close()

        for ds in ['tr', 'va', 'ts']:
            if ds not in seeds:
                raise ValueError("Missing %s dataset in seeds dict." % ds)

            if not np.iterable(seeds[ds]):
                raise ValueError("Seed list %s is not iterable." % ds)

            if len(self._datasets[ds]) != len(seeds[ds]):
                raise ValueError(("List of seed lists length doesn't match "
                                  " dataset %s length." % ds))

            for key in seeds[ds]:
                if len(seeds[ds][key]) == 0:
                    raise ValueError("Each example must have at least 1 seed.")

                for iseed,seed in enumerate(seeds[ds][key]):
                    if not np.iterable(seed):
                        raise ValueError("Each seed must be iterable.")
                    if len(seed) != self._ndim:
                        raise ValueError(("Number of seed coordinates doesn't "
                                          "match image dimensions."))
        self._seeds = seeds

    def _collect_scores(self):
        df = self._data_file
        tf = self._tmp_file

        if not hasattr(self, '_scores'):
            self._scores = {}
            for ds,key,iseed,seed in self._iter_tmp():
                if ds not in self._scores:
                    self._scores[ds] = {}
                if key not in self._scores[ds]:
                    self._scores[ds][key] = {}
                self._scores[ds][key][iseed] = []

        for ds,key,iseed,seed in self._iter_tmp():
            u   = tf["%s/%s/seed-%d/u" % (ds,key,iseed)][...]
            seg = df[key+"/seg"][...]
            score = self.score_func(u, seg)
            self._scores[ds][key][iseed].append(score)

        df.close()
        tf.close()

    def _get_mean_scores_at_iter(self, iter):
        mu = {}
        for ds in ['tr','va','ts']:
            mu[ds] = 0.0
            for key,iseed,seed in self._iter_seeds(ds):
                score = self._scores[ds][key][iseed][iter]
                mu[ds] += score / getattr(self, '_n'+ds)
        return [mu[ds] for ds in ['tr','va','ts']]

    def _scores_over_iters(self, dataset):
        assert dataset in ['tr','va','ts']
        names  = []
        scores = []
        for key,iseed,seed in self._iter_seeds(dataset):
            names.append("%s/%s/seed-%d" % (dataset,key,iseed))
            scores.append(self._scores[dataset][key][iseed])
        return np.vstack(scores), names


# This class is used privately within the stat learn level set model.
class fit_logger(logging.Logger):
    def __init__(self, file=None, stamp=True, stdout=True):
        fmt = '[%(asctime)s] %(levelname)-8s %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        now = datetime.datetime.now()
        timestamp = datetime.datetime.strftime(now, '%Y-%m-%d %H:%M:%S')

        if file is None:
            if stamp:
                base = "log-%s.txt" % timestamp
            else:
                base = "log.txt"
            file = os.path.join(os.path.curdir, base)
        else:
            if stamp:
                dir = os.path.dirname(file)
                base = os.path.basename(file)
                name,ext = os.path.splitext(base)
                base = "%s-%s%s" % (name, timestamp, ext)
                file = os.path.join(dir, base)

        self.file = file
        self.stamp = stamp
        self.stdout = stdout

        fhandler = logging.FileHandler(file, mode='w')
        fhandler.setFormatter(formatter)

        if self.stdout:
            shandler = logging.StreamHandler()
            shandler.setFormatter(formatter)

        logging.Logger.__init__(self, 'slls_fit_logger')
        self.setLevel(logging.DEBUG)

        self.addHandler(fhandler)

        if self.stdout: self.addHandler(shandler)

    def progress(self, msg, i, n):
        s = "(%%0%dd / %d) %s" % (len(str(n)), n, msg)    
        self.info(s % i)

def jaccard(u, seg, t=0.0):
    """
    Compute the Jaccard overlap score between `u > t` and `seg`.
    """
    H = u > t
    AND = (H & seg).sum() * 1.0
    OR  = (H | seg).sum() * 1.0

    if OR == 0:
        return 1.0
    else:
        return AND/OR
