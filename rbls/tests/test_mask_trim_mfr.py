import numpy as np
import mask_trim as mt

ntpr = int(1e4)

rs = np.random.RandomState(123)

X = rs.randn(ntpr, 3)
X /= np.linalg.norm(X,axis=1).reshape(ntpr,1)
thetas = np.arctan2(X[:,1], X[:,0])
phis = np.arccos(X[:,2])

radii = np.ones(ntpr)*25.0
seed  = np.ones(3, dtype=np.int)*50

mask = mt.mask_from_radii(thetas=thetas, phis=phis, seed=seed, radii=radii, n=101)

import matplotlib.pyplot as plt

plt.imshow(mask[:,:,50])
plt.show()
