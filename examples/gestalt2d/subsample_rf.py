import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SubsampleRF(RandomForestRegressor):

    max_samples = 3000

    def fit(self, X, y):

        self.n_outputs_ = 1

        defaults = {p: getattr(self, p) for p in self.estimator_params}
        self.estimators_ = []

        for i in range(self.n_estimators):
            print("Fitting tree {} / {}".format(i+1, self.n_estimators))
            est = type(self.base_estimator)(**defaults)
            inds = self._get_subsample_indices(X)
            est.fit(X[inds], y[inds])
            self.estimators_.append(est)

    def _get_subsample_indices(self, arr):
        return self.random_state.choice(np.arange(len(arr)), size=self.max_samples)


if __name__ == '__main__':
    import numpy as np

    rs = np.random.RandomState(123)

    x = np.linspace(0, 3.14159, 100)
    X = x[:,None]
    y = np.cos(x + 2*np.cos(2*x)) + rs.randn(x.shape[0])*0.1

    srf = SubsampleRF(n_estimators=100, random_state=rs)
    srf.max_samples = 10
    srf.fit(X, y)

    print(srf.score(X, y))
