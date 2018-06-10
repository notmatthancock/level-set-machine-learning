import os
import cPickle
import warnings
import numpy as np
import stats_recorder as sr
from datetime import datetime
from scipy.optimize import fmin_l_bfgs_b as fmin

class neural_network(object):
    """
    Single hidden layer neural network with single output unit that
    computes the identity.

    params: W, where w[i,j] = weight from input i to hidden unit j.
            b, where b[j] = bias into hidden unit j.
            z, where z[j] = weight from hidden unit j to output unit.
            c, single scalar bias for output unit.

    For a single vector input, the computation chain is:
    h = hidden_layer = tanh( dot(W.T, input) + b )
    output = dot(h, z) + c
    """
    def __init__(self, ninput, nhidden, rs=None, reg=0):
        """
        Parameters
        ----------
        ninput: int
            Number of input units.
 
        nhidden: int
            Number of hidden units.

        rs: numpy.random.RandomState, default=None
            Provide a RandomState object for reproducible results.

        reg: float, default=0
            The amount of L2 regularization of model parameters 
            to add to the loss function.
        """
        self.ni = ninput
        self.nh = nhidden
        self.rs = np.random.RandomState() if rs is None else rs

        # Initialize a stats_recorder to keep track of sample 
        # mean and standard deviation of observations.
        self.stats = sr.stats_recorder()

        self.reg = reg

        # This both intializes and randomizes.
        self.randomize_params()

    def __repr__(self):
        return "<neural_network ninput=%d, nhidden=%d>"%(self.ni, self.nh)

    def randomize_params(self, scale=0.1):
        """
        Randomize the model parameters using IID Gaussian random variables
        with variance `scale`.
        """
        self.W = scale * self.rs.randn(self.ni, self.nh)
        self.b = scale * self.rs.randn(self.nh)
        self.z = scale * self.rs.randn(self.nh)
        self.c = scale * self.rs.randn()

    def get_params(self, flat=False):
        """
        Parameters
        ----------
        flat: bool, default=False
            If True, the parameters are flattened into a single array.

        Returns
        -------
        params: list or array
            If `flat` is False (default), then the parameters are returned 
            as [W,b,z,c]. Otherwise, these are flattened into a single 
            array and returned.
        """
        if flat:
            return np.hstack([np.atleast_1d(p).flatten()
                              for p in self.get_params()])
        else:
            return [self.W, self.b, self.z, self.c]

    def set_params(self, W, b, z, c):
        """
        Set the parameter values to those providedin the arguments.
        """
        assert W.shape == (self.ni, self.nh), "W is wrong shape."
        assert b.shape == (self.nh,),         "b is wrong shape."
        assert z.shape == (self.nh,),         "z is wrong shape."
        assert np.isscalar(c),                "c should be a scalar."

        self.W = W; self.b = b
        self.z = z; self.c = c

    def save(self, base="./", fname=None, ts=True):
        """
        Pickle the model.
        """
        if fname is None and ts:
            fname = "nnet-model-%d-%d-%s.pkl" % (self.ni, self.nh,
                                                 datetime.now())
        elif fname is None and not ts:
            fname = "nnet-model-%d-%d.pkl" % (self.ni, self.nh)

        fpath = os.path.join(base, fname)

        with open(fpath, 'wb') as f: cPickle.dump(self, f)

    def predict(self, X):
        """
        Parameters
        ----------
        X: ndarray, shape=(nsamples, ninputs)
            Each row of `X` is an observation.

        Returns
        -------
        out: ndarray, shape=(nsamples,)
            out = dot(tanh(X,W)+b, z) + c
        """
        if self.stats.nobservations == 0:
            H = np.tanh(np.dot(X, self.W) + self.b)
        else:
            Z = (X-self.stats.mean) / self.stats.std
            H = np.tanh(np.dot(Z, self.W) + self.b)
            
        return np.dot(H, self.z) + self.c 

    def loss(self, X, y):
        """
        Compute the mean-squared-error loss between the network's 
        prediction of `X` and the values `y`.

        Parameters
        ----------
        X: ndarray, shape=(nsamples, ninput)
            The input array.

        y: ndarray, shape=(nsamples,)
            The response variable, i.e., the "correct" output values.

        Returns
        -------
        mse: float
            The mean-squared error between the network ouput with `X` as input
            and `y`.
        """
        diff = self.predict(X) - y
        loss = 0.5 * np.dot(diff, diff) / X.shape[0]
        if self.reg > 0:
            loss += 0.5*self.reg*(np.linalg.norm(self.W)**2  / self.ni +\
                             np.linalg.norm(self.z)**2) / self.nh
        return loss

    def lossgrad(self, X, y):
        """
        Return loss and [Dloss/Dparam for param in [W,b,z,c]] 
        """
        nsamples = X.shape[0]

        # Normalize if we are able.
        if self.stats.nobservations > 0:
            Z = (X-self.stats.mean) / self.stats.std
        else:
            Z = X

        # Compute the network's prediction.
        H = np.tanh(np.dot(Z, self.W) + self.b)
        o = np.dot(H, self.z) + self.c

        diff = o-y
        loss = 0.5 * np.dot(diff, diff) / nsamples

        # Add self.regularization penalty to loss.
        if self.reg > 0:
            loss += 0.5*self.reg*(np.linalg.norm(self.W)**2  / self.ni +\
                             np.linalg.norm(self.z)**2) / self.nh

        # Gradient wrt to bias term in "hidden => output".
        Dc = diff.mean()

        Htilde = diff.reshape(-1,1) * H

        # Gradient wrt linear coefs in "hidden => output".
        Dz = Htilde.mean(0)

        # Add self.regularization component to gradient.
        if self.reg > 0:
            Dz += self.reg * self.z / self.nh

        Htilde = Htilde * H

        # Gradient wrt bias terms in "input => hidden".
        Db = self.z * (Dc - Htilde.mean(0))

        xtilde = (diff.reshape(-1,1) * Z).mean(0)
        Htilde = Htilde * self.z / nsamples

        # Gradient wrt linear coefs in "input => hidden".
        DW = np.outer(xtilde, self.z) - np.dot(Z.T, Htilde)

        # Add self.regularization component to gradient.
        if self.reg > 0:
            DW += self.reg * self.W / self.nh / self.ni

        return loss, [DW, Db, Dz, Dc]

    def gradient_descent(self, Xtr, ytr, Xva=None, yva=None, step=0.1,
                         iters=10, update_stats=True, verbose=True,
                         ret_last=False, logger=None):
        """
        Run `iters` gradient descent steps.

        Parameters
        ----------
        Xtr: ndarray, shape=(nsamples, ninput)
            The training inputs -- examples by row.

        ytr: ndarray, shape=(nsamples,)
            The training outputs.

        Xva,yva: ndarrays, defaults=None
            These are the validation dataset analogues to `Xtr` and `ytr`.
            If given, then the MSE over this validation data is recorded.

        step: float 
            Step size for gradient descent steps.

        iters: int, default=10
            Number of gradient descent steps to run.

        ret_last: bool, default=False
            If True, only the loss on the last iteration is returned.

        update_stats: bool, default=True
            If True, the training data `Xtr` are used to update the
            running empirical mean and standard deviation for automatic
            variable normalization.

        verbose: bool, default=True
            Print the loss(es) for each iteration.

        Returns
        -------
        loss_tr[, loss_va]: ndarray[, ndarray], shapes=(iters+1,)
            The MSE over the training (and validation) sets. Note that
            these arrays are length `iters`+1 since the loss prior to the 
            first gradient descent step is recorded as the first entry.
        """
        if update_stats:
            self.stats.update(Xtr)
        if verbose:
            q = len(str(iters))
            pstr = "GD, ITER: %%0%dd, LOSSTR: %%.5f" % q
            if Xva is not None:
                pstr += ", LOSSVA: %.5f"

        loss_tr = np.zeros(iters+1)
        if Xva is not None:
            loss_va = np.zeros(iters+1)

        init_params = [self.W.copy(), self.b.copy(),
                       self.z.copy, self.c]

        for i in range(iters):
            loss_tr[i], G = self.lossgrad(Xtr, ytr)
            if Xva is not None:
                loss_va[i] = self.loss(Xva, yva)

            if verbose and Xva is None:
                print(pstr % (i, loss_tr[i]))
            elif verbose and Xva is not None:
                print(pstr % (i, loss_tr[i], loss_va[i]))

            self.W -= step*G[0]; self.b -= step*G[1]
            self.z -= step*G[2]; self.c -= step*G[3]

            # If a bad loss value is encountered, reduce the step size.
            # hard-coded reduction of 0.9 and max tries of 100
            bad_loss = False
            for loss_test_iter in range(100):
                # We need to check the loss after parameter update
                # to see if the step size caused a 'blow up'.
                ltr = self.loss(Xtr, ytr)
                lva = self.loss(Xva, yva)

                # Check if nan or if loss rose >= relative factor of 100%.
                bad_loss = np.isnan(ltr) or np.isnan(lva) or \
                           (ltr-loss_tr[i])/loss_tr[i] > 2.0

                if bad_loss:
                    if loss_test_iter == 99:
                        if logger is not None:
                            logger.warn(("Couldn't make a good descent step, "
                                         " returning initial parameters."))
                        self.set_params(*init_params)

                        if Xva is None:
                            if ret_last:
                                return loss_tr[i]
                            else:
                                return loss_tr[:i+1]
                        else:
                            if ret_last:
                                return loss_tr[i], loss_va[i]
                            else:
                                return loss_tr[:i+1], loss_va[:i+1]

                    # Correct the parameters back.
                    self.W += step*G[0]; self.b += step*G[1]
                    self.z += step*G[2]; self.c += step*G[3]

                    # Reduce the step by factor of 0.9. 
                    step *= 0.9

                    if logger is not None:
                        logger.warn(("Bad loss encountered, reducing step "
                                     "size to %.7f."%step))

                    # Perform grad desc update with smaller step.
                    self.W -= step*G[0]; self.b -= step*G[1]
                    self.z -= step*G[2]; self.c -= step*G[3]
                else:
                    # The loss was good, so we break the inner for loop
                    # to the gradient descent step (`i`) for loop.
                    break


        loss_tr[iters] = self.loss(Xtr, ytr)

        if Xva is not None:
            loss_va[iters] = self.loss(Xva, yva)

        if verbose and Xva is None:
            print(pstr % (i, loss_tr[i]))
        elif verbose and Xva is not None:
            print(pstr % (i, loss_tr[i], loss_va[i]))

        if Xva is None:
            if ret_last:
                return loss_tr[-1]
            else:
                return loss_tr
        else:
            if ret_last:
                return loss_tr[-1], loss_va[-1]
            else:
                return loss_tr, loss_va

def stop_early(loss_hist, hist_len=100, tol=0, dec=True):
    """
    Returns True when the linear trend over the `hist_len` most recent
    components of `loss_hist` is greater (or lesser if dec=False) than `eps`.
    """
    x = np.c_[np.ones(hist_len), np.arange(hist_len)+1]
        
    if len(loss_hist) >= hist_len:
        L = np.array(loss_hist[-hist_len:])
        (b,m),_,_,_ = np.linalg.lstsq(x, L, rcond=None)
        if (dec and m >= tol) or (not dec and m <= tol): 
            return True
    return False

class batch_queue():
    """
    Given data, this implements a queue of batches.
    """
    def __init__(self, batch_size):
        self.bs = batch_size

        # x and y queues
        self.xq = []
        self.yq = []

    def __len__(self):
        # only count the "last" element of the queue (the first element
        # of the list) if it is equal to the batch size.
        if len(self.xq) > 0:
            return len(self.xq[1:]) + (self.xq[0].shape[0] == self.bs)
        else:
            return 0

    def update(self, x, y):
        """
        Split x and y into batches and all the batches to the queue.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("`x` and `y` must have same number of examples.")

        # If the last element of the Q is not full batch, then update it.
        if len(self.xq) > 0 and len(self.xq[0]) < self.bs:
            n = len(self.xq[0])

            self.xq[0] = np.vstack([self.xq[0], x[:self.bs-n]])
            self.yq[0] = np.hstack([self.yq[0], y[:self.bs-n]])

            if self.bs-n >= len(x):
                return
            else:
                x = x[self.bs-n:]
                y = y[self.bs-n:]

        # Now split x and y into batches.
        if len(x) < self.bs: # Easy case, x and y don't make a full batch.
            self.xq.insert(0,x)
            self.yq.insert(0,y)
        else:
            nb = x.shape[0] // self.bs

            for i in range(nb + ((len(x) % self.bs) != 0)):
                self.xq.insert(0, x[i*self.bs : (i+1)*self.bs])
                self.yq.insert(0, y[i*self.bs : (i+1)*self.bs])


    def next(self):
        """
        Return the next batch (x,y) from the queue.
        """
        if len(self) > 0:
            return self.xq.pop(), self.yq.pop()
        else:
            raise RuntimeError("The queue has no batches to offer.")
