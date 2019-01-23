import os
import shutil

import skfmm

from level_set_machine_learning.core.fit_job_handler import FitJobHandler
from level_set_machine_learning.gradient import masked_gradient as mg
from level_set_machine_learning.score_functions import jaccard
from level_set_machine_learning.feature.feature_map import FeatureMap
from level_set_machine_learning.initializer.initializer_base import (
    InitializerBase)


class LevelSetMachineLearning:

    def __init__(self, features, initializer, scorer=jaccard, band=3):
        """
        Initialize a level set machine learning object

        Note
        ----
        The parameters required at initialization are those that are necessary
        both at train- and run-time.

        Parameters
        ----------
        features: List(BaseFeature)
            A list of image or shape features that derive from BaseImageFeature
            or BaseShapeFeature

        initializer: class
            Provides the initial segmentation guess, given an image.
            A subclass of
            :class:`level_set_machine_learning.initializer.InitializerBase`

        model_class: class
            The regression model for modeling the level set velocity

        model_kwargs: dict
            The keyword arguments to supply :code:`model_class` on
            instantiation

        scorer: function, default=jaccard
            Has signature::
                
                scorer(u, seg)

            where `u` is a level set iterate and `seg` is the ground-truth
            segmentation. The default (None) uses the jaccard overlap 
            between `u > 0` and `seg`.

        band: float, default=3
            The "narrow band" distance.

        """
        # Create the feature map comprising the given features
        self.feature_map = FeatureMap(features=features)

        # Validate the level set initializer
        if not isinstance(initializer, InitializerBase):
            msg = ("`initializer` should be a class derived from "
                   "`level_set_machine_learning.initializer.InitializerBase`")
            raise ValueError(msg)
        self.initializer = initializer

        self.scorer = scorer
        self.band = band

        self.fit_job_handler = None
        self._is_fitted = False

    def _update_level_sets(self):
        df = self._data_file()
        tf = self._tmp_file_write_lock()

        model = self.models[-1]

        # Loop over all indices in the validation dataset.
        for ds, key, iseed, seed in self._iter_tmp():
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

            # Compute features.
            features = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

            # Compute approximate velocity from features.
            nu = np.zeros_like(u)
            nu[mask] = model.predict(features[mask])

            # Compute gradient magnitude using upwind Osher/Sethian method.
            gmag = mg.gradient_magnitude_osher_sethian(u, nu, mask=mask, dx=dx)

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
                    # The distance transform might not yield a mask
                    # if band is very large or the object is very large.
                    mask = np.ones(dist.shape, dtype=np.bool)

            # Update the data in the hdf5 file.
            tf["%s/%s/seed-%d/u"    % (ds, key, iseed)][...] = u
            tf["%s/%s/seed-%d/dist" % (ds, key, iseed)][...] = dist
            tf["%s/%s/seed-%d/mask" % (ds, key, iseed)][...] = mask

        df.close()
        tf.close()
        self._tmp_file_write_unlock()

    def _validation_dummy_update(self, nnet):
        """
        Do a "dummy" update over the validation dataset to 
        record segmentation scores.
        """
        df = self._data_file()
        tf = self._tmp_file(mode='r')

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
                gmag = mg.gradient_magnitude_osher_sethian(u, nu, mask=mask, dx=dx)

                # Here's the dummy update.
                utmp = u.copy()
                utmp[mask] += self.step*nu[mask]*gmag[mask]

            mu += self.scorer(utmp, seg) * 1.0 / self._nva

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
        """ Featurize all the images in the dataset given by the argument
        
        dataset: str
            Should be in ['tr','va','ts']

        balance: bool
            Balance by neg/pos of target value.

        random_state: numpy.RandomState
            To make results reproducible. The random state
            should be passed here rather than using the `self._random_state`
            attribute, since in the multiprocessing setting we can't
            rely on `self._random_state`.

        Returns
        -------
        X, y: ndarray (n_examples, n_features), ndarray (n_examples,)
            X is a matrix storing the feature vector in each row, whereas
            y is a vector storing the corresponding target value for
            each feature vector (i.e., the ground-truth signed distance
            value at the spatial coordinate for which the feature vector
            was computed).
        """
        assert dataset in ['tr','va','ts']

        ###########################################################
        # Precompute total number of feature vector examples

        # Count the number of points collected
        count = 0

        if balance:
            balance_masks = []

        for key, iseed, seed in self._iter_seeds(dataset):

            with self._tmp_file(mode='r') as tf:
                mask = tf["%s/%s/seed-%d/mask" % (dataset, key, iseed)][...]

            if balance:

                with self._data_file() as df:
                    target = df[key+"/dist"][...]

                balance_mask = self._balance_mask(target[mask], rs=rs)
                balance_masks.append(balance_mask)
                count += balance_mask.sum()

            else:
                count += mask.sum()

        X = np.zeros((count, self.feature_map.nfeatures))
        y = np.zeros((count,))

        index = 0

        ###########################################################
        # Compute the feature vectors and place them in
        # the feature matrix

        for i, (key, iseed, seed) in enumerate(self._iter_seeds(dataset)):

            with self._data_file() as df:
                img    = df[key+"/img"][...]
                target = df[key+"/dist"][...]
                dx     = df[key].attrs['dx']

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            with self._tmp_file(mode='r') as tf:
                u    = tf["%s/%s/seed-%d/u"    % (dataset, key, iseed)][...]
                dist = tf["%s/%s/seed-%d/dist" % (dataset, key, iseed)][...]
                mask = tf["%s/%s/seed-%d/mask" % (dataset, key, iseed)][...]

            if not mask.any():
                continue # Otherwise, repeat the loop until a non-empty mask.

            # Compute features.
            features = self.feature_map(u, img, dist=dist, mask=mask, dx=dx)

            if balance:
                bmask = balance_masks[i]
                next_index = index + target[mask][bmask].shape[0]
                X[index:next_index] = features[mask][bmask]
                y[index:next_index] =   target[mask][bmask]
            else:
                next_index = index + mask.sum()
                X[index:next_index] = features[mask]
                y[index:next_index] =   target[mask]

            index = next_index

        return X, y

    def _featurize_random_image(self, dataset, balance, rs):
        """
        dataset: str
            Should be in ['tr','va','ts']

        balance: bool
            Balance by neg/pos of target value.

        random_state: numpy.RandomState
            To make results reproducible. The random state
            should be passed here rather than using the `self._random_state`
            attribute, since in the multiprocessing setting we can't
            rely on `self._random_state`.
        """
        assert dataset in ['tr','va','ts']
        who = ['tr','va','ts'].index(dataset)

        while True:
            # Get a random image from the appropriate dataset.
            key   = rs.choice(self._seeds[dataset].keys())
            iseed = rs.choice(len(self._seeds[dataset][key]))

            with self._data_file() as df:
                img    = df[key+"/img"][...]
                target = df[key+"/dist"][...]
                dx     = df[key].attrs['dx']

            if self._fopts_normalize_images:
                img = (img - img.mean()) / img.std()

            with self._tmp_file(mode='r') as tf:
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


    def _fit_model(self):
        """
        Fit a regression model for advancing the level set.
        """
        self._iter_dir = os.path.join(self._models_path, 
                                      "iter-%d" % self._iter)
        os.mkdir(self._iter_dir)

        self._logger.info('Featurizing training images.')

        # Get input and output variables
        X, y = self._featurize_all_images('tr', balance=True,
                                          rs=self._random_state)
        # Create and fit the model
        model = self.model(**self.model_kwargs)
        model.fit(X, y)
        self.models.append(model)

        #np.save(os.path.join(self._fopts_tmp_dir, 'X.npy'), X)
        #np.save(os.path.join(self._fopts_tmp_dir, 'y.npy'), y)

        #self._logger.info("Fitting random forest.")

        ## Fit in a new process for memory release
        #p = mp.Process(target=self._fit_rf, args=(), name='rf fit')

        #p.start()
        #p.join()

        #if p.exitcode != 0:
        #    msg = ("An error occured during tree training (%s)."
        #            % p.name)
        #    self._logger.error(msg)
        #    raise RuntimeError(msg)

        # Remove the temporary neural net files if necessary.
        if self._fopts_remove_tmp:
            self._logger.info("Removing tmp files at %s." % self._iter_dir)
            shutil.rmtree(self._iter_dir)
        del self._iter_dir

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

    def fit(self, data_filename, regression_model_class,
            regression_model_kwargs, imgs=None, segs=None, dx=None,
            normalize_imgs_on_convert=True, datasets_split=(0.6, 0.2, 0.2),
            subset_size=None, step=None, temp_data_dir=os.path.curdir,
            validation_history_len=5, validation_history_tol=0.0,
            max_iters=100, random_state=None):
        """ Fit a level set machine learning segmentation model

        Parameters
        ----------
        data_filename: str
            The filename of the hdf5 file of converted data or the filename
            of an existing hdf5 file with required structure. If the
            file `h5_file` does not exist, then `imgs` and `segs` *must* be
            provided.

        regression_model_class: class
            The regression model class. At each iteration of level set
            evolution, a new instance of this class is instantiated to create
            the respective velocity model for the iteration

        regression_model_kwargs: dict
            A dictionary of keyword arguments used to instantiate the provided
            :code:`regression_model_class`. For example,
            :code:`{'n_estimators': 100, 'random_state': RandomState(123)}`
            might be used with corresponding :code:`RandomForestRegressor`
            model class.

        imgs: list of ndarray, default=None
            The list of images to be used if :code:`data_file` points to a
            (currently) non-existent hdf5 file. The default of None
            assumes that `data_file` points to an existing hdf5 file

        segs: list of ndarray
            The list of segmentation to be used if :code:`data_file` points to
            a (currently) non-existent hdf5 file. The default of None
            assumes that `data_file` points to an existing hdf5 file

        dx: list of ndarray, default=None
            The list of respective delta terms for the provided images.
            :code:`dx[i]` should be a list or array corresponding to the
            i'th image with length corresponding to the image dimensions.
            The default of None assumes isotropicity with a value of 1.

        normalize_imgs_on_convert: bool, default=True
            If True, then the provided images are individually normalized
            by their means and standard deviations on conversion to hdf5

        datasets_split: 3-tuple, default=(0.6, 0.2, 0.2)
            The items in the tuple correspond to the training, validation, and
            testing datasets. If each item is a float, then the provided data
            is randomly split with probabilities of respective values.
            Alternatively, each item in the 3-tuple may be a list of integer
            indexes explicitly prescribing each example to a specific dataset
            for fitting.

        subset_size: int, default=None
            If datasets are randomly partitioned, then the full dataset
            is first down-sampled to be `subset_size` before partitioning

        step: float, default=None
            The step size for updating the level sets, i.e., the "delta t"
            term in the discretization of :math:`u_t = \\nu \\| Du \\|`.

            The default None determines the step size automatically
            by using the reciprocal of the maximum ground truth speed values
            in the narrow band of the level sets in the training data at
            the first iterate (i.e., at initializer). The CFL condition
            is satisfied by taking the maximum speed over ALL spatial
            coordinates, but we attempt to avoid this prohibitively
            small step size by assuming that the maximum speed observed
            will be in the first iteration (i.e., that u0 is farthest
            from the ground-truth).

        temp_data_dir: str, default=os.path.curdir
            Where to store the temporary data that is created during the
            fitting process. This data is removed after fitting and includes,
            for example, the per-iteration level set values

        validation_history_len: int, default=5
            The number of past iterations back from the current to check
            scores over the validation dataset to monitor progress from

        validation_history_tol: float, default=0.0
            If the linear trend in scores over the last
            :code:`validation_history_len` iterations is less than this
            value, then we exit early.

        max_iters: int, default=100
            The fixed maximum number of iterations

        random_state: numpy.random.RandomState, default=None
            Provide for reproducible results.

        """
        # Dirty tricks to initialize the fit handler...
        kwargs = locals()
        kwargs['model'] = kwargs.pop('self')
        self.fit_job_handler = FitJobHandler(**kwargs)

        # Set up the level sets according the initializer functions
        self.fit_job_handler.initialize_level_sets()

        # Compute and store scores at initialization (iteration = 0)
        #self.fit_job_handler.compute_and_store_scores()

        #while self.fit_job_handler.can_continue():

        #    self.fit_job_handler.self.fit_regression_model()
        #    self.fit_job_handler.update_level_sets()
        #    self.fit_job_handler.compute_and_store_scores()
        #    self.save()

        ## Remove temp data and toss models after max of validation data
        #self.fit_job_handler.clean_up()  # set model._is_fitted = True

        #self.save()

    def segment(self, img, seg=None, dx=None, verbose=True):
        """
        Segment `img`.

        Parameters
        ----------
        img: ndarray
            The image

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
            raise RuntimeError("This model has not been fit yet")

        iters = len(self.models)
        dx = np.ones(img_.ndim) if dx is None else dx

        if dx.shape[0] != img_.ndim:
            raise ValueError("`dx` has incorrect number of elements.")

        u = np.zeros((iters+1,) + img_.shape)
        u[0], dist, mask = self.initializer(img_, self.band, dx=dx)

        if seg is not None:
            scores = np.zeros((iters+1,))
            scores[0] = self.scorer(u[0], seg)

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

                gmag = mg.gradient_magnitude_osher_sethian(u[i], nu, mask=mask, dx=dx)
                    
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
                scores[i+1] = self.scorer(u[i + 1], seg)

            if verbose and seg is None:
                print(pstr % (i+1))
            elif verbose and seg is not None:
                print(pstr % (i+1, scores[i+1]))

        if seg is None:
            return u
        else:
            return u, scores
    
    def _collect_scores(self):
        df = self._data_file()
        tf = self._tmp_file(mode='r')

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
            score = self.scorer(u, seg)
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
