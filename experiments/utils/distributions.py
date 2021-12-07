import numpy as np
from sklearn.utils import shuffle

def gaussian_dist(dim, corr_factor, n_samples):
    '''
    Generates samples from two gaussian random variables X_a and X_b of 
    dimension dim with Corr(X_a^i, X_b^j) = delta_ij * corr_factor, returning the 
    theoretical mutual information MI(X_a, X_b).
    '''
    joint_mean = np.zeros(2 * dim)
    identity = np.identity(dim)
    joint_cov = np.concatenate([np.concatenate([identity, corr_factor * identity], axis=1),
                                np.concatenate([corr_factor * identity, identity], axis=1)], 
                                axis=0)
    np.fill_diagonal(joint_cov, 1)
    theoretical_mi = -(1/2) * np.log2(np.linalg.det(joint_cov))
    X_joint = np.random.multivariate_normal(joint_mean, joint_cov, size=n_samples)
    X_a = X_joint[:, :dim]
    X_b = X_joint[:, dim:]
    return X_joint, X_a, X_b, theoretical_mi


def t_dist(dim, corr_factor, n_samples, freedom_deg=2):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    N = freedom_deg
    mu = np.zeros(2 * dim)
    M = n_samples
    g = np.tile(np.random.gamma(N/2.,2./N,M),(2*dim,1)).T
    X_joint, _, _, _ = gaussian_dist(dim, corr_factor, n_samples)
    X_joint = mu + X_joint / np.sqrt(g)
    X_a = X_joint[:, :dim]
    X_b = X_joint[:, dim:]
    return X_joint, X_a, X_b, None

def gaussian_mixture_dist(rot, n_samples):
    '''
    Generates samples from a mixture of 4 Gaussian distributions on each quadrant of the cartesian system
    rotated by $rot$ radians around the origin.
    '''
    std = 0.05
    values = np.arange(n_samples, dtype=int)
    n_samples_bin = np.histogram(values, bins=4)[0]
    X1, Y1 = np.random.multivariate_normal([4,4], [[std,0],[0,std]], n_samples_bin[0]).T
    X2, Y2 = np.random.multivariate_normal([4,2], [[std,0],[0,std]], n_samples_bin[1]).T
    X3, Y3 = np.random.multivariate_normal([2,4], [[std,0],[0,std]], n_samples_bin[2]).T
    X4, Y4 = np.random.multivariate_normal([2,2], [[std,0],[0,std]], n_samples_bin[3]).T
    X = np.expand_dims(np.concatenate([X1,X2,X3,X4]), axis=1)
    Y = np.expand_dims(np.concatenate([Y1,Y2,Y3,Y4]), axis=1)
    Xrot = X * np.cos(rot) - Y * np.sin(rot)
    Yrot = X * np.sin(rot) + Y * np.cos(rot)
    Xrot, Yrot = shuffle(Xrot, Yrot)
    return Xrot, Yrot