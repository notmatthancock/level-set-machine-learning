import os, shutil, time, warnings, pickle, h5py
import sys, datetime, logging
import numpy as np
import multiprocessing as mp
import skfmm

from feature_maps import feature_map_base
from init_funcs import init_func_base
from utils.data import splitter

#from utils.featurize import featurize
import utils.masked_grad as mg
from neural_network import neural_network as nn

class RBLS(object):
    def __init__(self, data_file, feature_map, init_func, score_func=None,
                 step='auto', band=3, rs=None): 
        """
        Initialize a RBLS object. The parameters required at initialization 
        are those that can be used in *both* the training and run-time
        phases of the algorithm.

        Parameters
        ----------
        data_file: str
            This string should be of the form::
                
                some/path/to/data.h5

            where `data.h5` is formatted as specified in the 
            :meth:`rbls.utils.tohdf5` module. `data.h5` contains the 
            image and ground truth data.

        feature_map: feature map class
            See :class:`rbls.feature_maps.feature_map_base`.

        init_func: init func class
            See :class:`rbls.init_funcs.init_func_base`.

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
        self._data_file = data_file

        with self._get_data_file() as df:
            self._n_examples = len(df)

        if not isinstance(feature_map, feature_map_base):
            raise ValueError(("`feature_map` should be a class derived from"
                              " `rbls.feature_maps.feature_map_base`.")) 
        self.feature_map = feature_map

        if not isinstance(init_func, init_func_base):
            raise ValueError(("`init_func` should be a class derived from"
                              " `rbls.init_funcs.init_func_base`.")) 
        self.init_func = init_func

        self.score_func = jaccard if score_func is None else score_func
        self.band = band

        if step != 'auto' and not isinstance(step, float):
            raise ValueError("`step` must be 'auto' or float.")
        self.step = step

        if rs is not None:
            self._rs = rs
        else:
            warnings.warn(("No RandomState provided. Results will not be"
                           " reproducible."))

        self._is_fitted    = False
        self._fit_opts_set = False
        self._net_opts_set = False

    def _get_data_file(self):
        return h5py.File(self._data_file, 'r')
    def _get_tmp_file(self):
        return h5py.File(os.path.join(self._fopts_tmp_dir, "tmp.h5"))

    def _initialize(self):
        df = self._get_data_file()
        tf = self._get_tmp_file()

        step = np.inf

        for i in range(self._n_examples):
            # Run the initialization function.
            u0,dist,mask = self.init_func(df["%d/img"%i][...], self.band,
                                          dx=df["%d"%i].attrs['dx'])

            # "Auto" step: only use training or validation sets.
            if i in self._inds[0] or i in self._inds[1]:
                # Compute the maximum velocity for the i'th example
                mx = np.abs(df["%d/dist"%i][mask]).max()

                # Create the candidate step size for this example.
                tmp = np.min(df["%d"%i].attrs['dx']) / mx

                # Assign tmp to step if it is the smallest observed so far.
                step = tmp if tmp < step else step
            
            # Create a group for the i'th example.
            g = tf.create_group("%d"%i)

            # The group consists of the current "level set field" u, the 
            # signed distance transform to `u` (only in the narrow band), and
            # the boolean mask indicating the narrow band region.
            g.create_dataset("u",    data=u0,   compression='gzip')
            g.create_dataset("dist", data=dist, compression='gzip')
            g.create_dataset("mask", data=mask, compression='gzip')

        if self.step == 'auto':
            self.step = step
            self._logger.info("Computed auto step is %.7f." % self.step)
        elif self.step != 'auto' and self.step > step:
            self._logger.warn("Computed step is %.7f but given step is %.7f."
                                % (step, self.step))
        df.close()
        tf.close()

    def _update_level_sets(self):
        df = self._get_data_file()
        tf = self._get_tmp_file()

        nnet = self.models[-1]

        # Loop over all indices in the validation dataset.
        for i in range(self._n_examples):
            img = df["%d/img"%i][...]
            dx = df["%d"%i].attrs['dx']

            u = tf["%d/u"%i][...]
            dist = tf["%d/dist"%i][...]
            mask = tf["%d/mask"%i][...]

            # Don't update if the mask is empty.
            if not mask.any(): continue

            # Check if level set vanished and set mask to zeros if so.
            if (u > 0).all() or (u < 0).all():
                tf["%d/mask"%i] = np.zeros_like(mask)
                continue

            # Compute features.
            F = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

            # Compute approximate velocity from features.
            nu_hat = np.zeros_like(u)
            nu_hat[mask] = nnet.predict(F[mask])

            # Compute gradient magnitude using upwind Osher/Sethian method.
            gmag = mg.gmag_os(u, nu_hat, mask=mask, dx=dx)

            # Here's the actual level set update.
            u[mask] += self.step*nu_hat[mask]*gmag[mask]

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
            tf["%d/u"%i][...] = u
            tf["%d/dist"%i][...] = dist
            tf["%d/mask"%i][...] = mask

        df.close()
        tf.close()

    def _validation_dummy_update(self, nnet):
        """
        Do a "dummy" update over the validation dataset to 
        record segmentation scores.
        """
        df = self._get_data_file()
        tf = self._get_tmp_file()

        mu = 0.0; nva = float(len(self._inds[1]))

        # Loop over all indices in the validation dataset.
        for i in self._inds[1]:
            img = df["%d/img"%i][...]
            seg = df["%d/seg"%i][...]
            dx = df["%d"%i].attrs['dx']

            u = tf["%d/u"%i][...]
            dist = tf["%d/dist"%i][...]
            mask = tf["%d/mask"%i][...]

            # Compute features.
            F = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)
            # Compute approximate velocity from features.
            nu_hat = np.zeros_like(u)
            nu_hat[mask] = nnet.predict(F[mask])
            # Compute gradient magnitude using upwind Osher/Sethian method.
            gmag = mg.gmag_os(u, nu_hat, mask=mask, dx=dx)

            # Here's the dummy update.
            utmp = u.copy()
            utmp[mask] = u[mask] + self.step*nu_hat[mask]*gmag[mask]

            mu += self.score_func(utmp, seg) / nva

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

    def _featurize_image(self, dataset, balance, rs):
        """
        dataset: str
            Should be in ['tr','va','ts']

        balance: bool
            Balance by neg/pos of target value.

        rs: numpy.RandomState
            To make results reproducible.
        """
        who = ['tr','va','ts'].index(dataset)

        while True:
            # Get a random image from the appropriate dataset.
            i = rs.choice(self._inds[who])

            df = self._get_data_file()
            img = df["%d/img"%i][...]
            target = df["%d/dist"%i][...]
            dx = df["%d"%i].attrs['dx']
            df.close()

            tf = self._get_tmp_file()
            u = tf["%d/u"%i][...]
            dist = tf["%d/dist"%i][...]
            mask = tf["%d/mask"%i][...]
            tf.close()

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
                    x,y = self._featurize_image(dataset='tr', 
                                                balance=balance, rs=rs)
                    bqtr.update(x,y)

                while len(bqva) == 0:
                    x,y = self._featurize_image(dataset='va',
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
                                                    verbose=False)
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
        df = self._get_data_file()
        tf = self._get_tmp_file()

        mu = 0.0
        n = float(len(self._inds[0]))
        balance = self._nopts_balance

        for i in self._inds[0]:
            mask = tf["%d/mask"%i][...]
            targ = df["%d/dist"%i][...]

            if not balance:
                mu += mask.sum() / n
            else:
                npos = (targ[mask] > 0).sum()
                nneg = (targ[mask] < 0).sum()
                
                if 0 < nneg < npos:
                    mu += 2*nneg / n
                elif 0 < npos < nneg:
                    mu += 2*npos / n
                else:
                    mu += max(npos, nneg) / n

        self._auto_batch_size = int(mu) * self._nopts_batch_size
        self._logger.info("Computed auto batch size is %d."
                                 % self._auto_batch_size)

        df.close()
        tf.close()

    def _compute_auto_batches_per_epoch(self):
        df = self._get_data_file()
        tf = self._get_tmp_file()

        total = 0.0
        balance = self._nopts_balance

        for i in self._inds[0]:
            mask = tf["%d/mask"%i][...]
            targ = df["%d/dist"%i][...]

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

        self._logger.info("Computed auto batches per epoch is %d."
                                 % self._auto_batches_per_epoch)

        df.close()
        tf.close()
        
    def _fit_model(self):
        """
        Obtain the best model over all hidden units tested and over all
        number of parameter initializations.
        """
        self._iter_dir = os.path.join(self._models_path, 
                                      "iter-%d" % self._iter)
        os.mkdir(self._iter_dir)

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
                msg = "An error occured during nnet training (%s)." % p.name
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

        # Remove the temporary neural net files if necessary.
        if self._fopts_remove_tmp:
            self._logger.info("Removing tmp files at %s." % self._iter_dir)
            shutil.rmtree(self._iter_dir)
        del self._iter_dir

    def _get_mean_scores(self, iter):
        return [self.scores[iter][self._inds[i]].mean()
                    for i in range(3)]

    def fit(self):
        """
        Run `set_fit_options` and `set_net_opts` before calling `fit`.
        """
        if not self._fit_opts_set:
            raise RuntimeError("Run `set_fit_options` before fitting.")
        if not self._net_opts_set:
            raise RuntimeError("Run `set_net_opts` before fitting.")
        if self._is_fitted:
            raise RuntimeError("This model has already been fitted.")

        
        self._logger.info("Initializing.")
        self._initialize()
        self._iter = 0

        self._logger.info("Collecting scores.")
        self._collect_scores()
        self._logger.progress("Scores => TR: %.4f VA: %.4f TS %.4f."
                              % tuple(self._get_mean_scores(self._iter)),
                              self._iter, self._fopts_maxiters)

        self._models_path = os.path.join(self._fopts_tmp_dir, "models")
        os.mkdir(self._models_path)

        maxiters = self._fopts_maxiters

        for self._iter in range(1, maxiters+1):
            self._logger.progress("Beginning iteration.",
                                  self._iter, maxiters)
            iter_start = time.time()

            self._logger.info("Fitting nnets.")
            self._fit_model()

            self._logger.progress("Updating level sets.", self._iter, maxiters)
            self._update_level_sets()

            self._logger.progress("Collecting scores.", self._iter, maxiters)
            self._collect_scores()
            self._logger.progress("Scores => TR: %.4f VA: %.4f TS %.4f."
                                  % tuple(self._get_mean_scores(self._iter)),
                                  self._iter, maxiters)

            iter_stop = time.time()
            iter_time = (iter_stop-iter_start) / 3600.

            self._logger.progress(("End of iteration. Ellapsed time: "
                                   "%.3f hours.") % iter_time,
                                  self._iter, maxiters)

            self._logger.info("Saving model to %s." % self._fopts_save_file)

            # Swap the logger before saving the model since
            # file objects don't pickel.
            logger = self._logger
            self._logger = None

            # Save the model. 
            with open(self._fopts_save_file, 'w') as f:
                pickle.dump(self, f)

            # Put the logger back.
            self._logger = logger


        if self._fopts_remove_tmp:
            self._logger.info("Removing temporary files at %s." 
                                    % self._fopts_tmp_dir)
            shutil.rmtree(self._fopts_tmp_dir)


        self._is_fitted = True
        self._logger.info("Fitting complete.")

        self._logger.info("Saving model to %s." % self._fopts_save_file)

        # We have to remove the logger before saving the model
        # since we can't pickle file objects.
        for h in self._logger.handlers: h.close()
        del self._logger

        # Save the model. 
        with open(self._fopts_save_file, 'w') as f:
            pickle.dump(self, f)


    def set_fit_options(self, inds=None, save_file=None,
                        tmp_dir=None, remove_tmp=False,
                        maxiters=100, va_hist_len=5, va_hist_tol=0.0,
                        logfile=None, logstamp=True, logstdout=True):
        """
        Parameters
        ----------
        inds: tuple, len=3, default=None
            This should be a tuple of 3 lists. The lists contain the
            indices to be used as training, validation, and testing, 
            respectively. The default (None) creates a random split
            of 60/20/20% using the :meth:`rbls.utils.data.splitter.split`
            routine.

        save_file: str, default=None
            The model (the total RBLS object) will be pickled to this path.
            The default (None) uses the file `rbls_model.pkl` at the current
            working directory.

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
        # This is hackish. It just sets most the arguments to member variables.
        L = locals(); L.pop('self'); L.pop('inds')
        for l in L: setattr(self, "_fopts_%s"%l, L[l])

        # Validate and store the training/validation/testing indices.
        if inds is not None:
            self._validate_inds(inds)
        else:
            inds = splitter.split(self._n_examples, rs=self._rs)

        if self._fopts_save_file is None:
            self._fopts_save_file = os.path.join(os.path.curdir,
                                                 "rbls_model.pkl")

        if self._fopts_tmp_dir is None:
            self._fopts_tmp_dir = os.path.join(os.path.curdir, 'tmp')
        else:
            self._fopts_tmp_dir = os.path.join(self._fopts_tmp_dir, 'tmp')

        os.mkdir(self._fopts_tmp_dir)

        self._inds = inds
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

    def segment(self, img, seg=None, dx=None, verbose=True):
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

        iters = len(self.models)
        dx = np.ones(img.ndim) if dx is None else dx

        if dx.shape[0] != img.ndim:
            raise ValueError("`dx` has incorrect number of elements.")

        u = np.zeros((iters+1,) + img.shape)
        u[0],dist,mask = self.init_func(img, self.band)

        if seg is not None:
            scores = np.zeros((iters+1,))
            scores[0] = self.score_func(u[0], seg)

        nu = np.zeros(img.shape)

        if verbose:
            if seg is None:
                pstr = "Iter: %02d"
                print(pstr % 0)
            else:
                pstr = "Iter: %02d, Score: %0.5f"
                print(pstr % (0, scores[0]))

        for i in range(iters):
            if not mask.any(): continue
            if (u > 0).all() or (u < 0).all():
                mask = np.zeros_like(mask)
                continue

            # Compute the features, and use the model to predict speeds.
            mem = 'create' if i==0 else 'use'
            features = self.feature_map(u[i], img, dist=dist,
                                        mask=mask, memoize=mem,
                                        dx=dx)
            nu[mask] = self.models[i].predict(features[mask])

            gmag = mg.gmag_os(u[i], nu, mask=mask, dx=dx)
                
            # Update the level set.
            u[i+1] = u[i].copy()
            u[i+1][mask] = u[i][mask] + self.step*nu[mask]*gmag[mask]

            # Update the signed distance function for phi.
            dist = skfmm.distance(u[i], narrow=self.band, dx=dx)

            if hasattr(dist, 'mask'):
                mask = ~dist.mask
                dist =  dist.data
            else:
                if u.ravel()[0] > 0:
                    mask = np.ones(u[i].shape, dtype=np.bool)
                    warnings.warn("`u` is all positive.")
                else:
                    mask = np.zeros(u[i].shape, dtype=np.bool)
                    warnings.warn("`u` is all negative.")

            if seg is not None:
                scores[i+1] = self.score_func(u[i], seg)

            if verbose and seg is None:
                print(pstr % i)
            elif verbose and seg is not None:
                print(pstr % (i+1, scores[i+1]))

        if seg is None:
            return u
        else:
            return u, scores

    
    def _validate_inds(self, inds):
        if len(inds) != 3:
            raise ValueError("`inds` should be length 3.")
        if not all([isinstance(i, list) for i in inds]):
            raise ValueError("`inds` should be tuple of three index lists.")
        if len(set(inds[0]).intersection(set(inds[1]))) > 0:
            raise ValueError("Training set indices overlap with validation.")
        if len(set(inds[0]).intersection(set(inds[2]))) > 0:
            raise ValueError("Training set indices overlap with testing.")
        if len(set(inds[1]).intersection(set(inds[2]))) > 0:
            raise ValueError("Validation set indices overlap with testing.")
        if set(inds[0] + inds[1] + inds[2]) - set(range(self._n_examples)) > 0:
            raise ValueError("`inds` does not include all the data.")

    def _collect_scores(self):
        df = self._get_data_file()
        tf = self._get_tmp_file()

        if not hasattr(self, 'scores'):
            self.scores = np.zeros((self._fopts_maxiters+1, self._n_examples))

        for i in range(self._n_examples):
            s = self.score_func(tf["%d/u"%i][...], df["%d/seg"%i][...])
            self.scores[self._iter, i] = s

        df.close()
        tf.close()

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
        self.stamp = True
        self.stdout = True

        fhandler = logging.FileHandler(file, mode='w')
        fhandler.setFormatter(formatter)

        if stdout:
            shandler = logging.StreamHandler()
            shandler.setFormatter(formatter)

        logging.Logger.__init__(self, 'rbls_fit_logger')
        self.setLevel(logging.DEBUG)

        self.addHandler(fhandler)

        if stdout: self.addHandler(shandler)

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
