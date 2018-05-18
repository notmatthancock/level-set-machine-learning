import h5py
import threshold_init as ti
import numpy as np
import socket
from tqdm import tqdm

J = lambda A,B: (A&B).sum()*1.0 / (A|B).sum()

xtra = 'Data/' if ('asus' not in socket.gethostname()) else ''

def runall(sigma, pr, ds='va'):
    scores = np.zeros(112)

    sigma = np.array(sigma)

    hf = h5py.File('/home/matt/'+xtra+'LIDC/3d/no-interp/dataset.h5', 'r')

    for n in range(112):
        img = hf[ds+'/%03d/img'%n][...]
        seg = hf[ds+'/%03d/seg'%n][...]
        dij = hf[ds+'/%03d'%n].attrs['delta_ij']
        dk  = hf[ds+'/%03d'%n].attrs['delta_k']
        com = hf[ds+'/%03d'%n].attrs['com']
        aid = hf[ds+'/%03d'%n].attrs['aids']

        B = ti.threshold_init(img, sigma=sigma, pr=pr,
                              spacing=[dij,dij,dk], alert_small=0,
                              seed=com, only_seg=True, adjust_sigma=1)

        scores[n] = J(B,seg)

    hf.close()

    #np.save("scores.npy", scores)
    return scores

def runone(n):
    sigma = np.r_[4,4,1.0]

    img = hf['ts/%03d/img'%n][...]
    seg = hf['ts/%03d/seg'%n][...]
    dij = hf['ts/%03d'%n].attrs['delta_ij']
    dk  = hf['ts/%03d'%n].attrs['delta_k']
    com = hf['ts/%03d'%n].attrs['com']
    aid = hf['ts/%03d'%n].attrs['aids']

    B = ti.threshold_init(img, sigma=sigma, pr=70.0, spacing=[dij,dij,dk],
                          seed=com, only_seg=True)

    score = J(B,seg)
    return locals()
