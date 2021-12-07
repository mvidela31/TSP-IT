import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, gamma

class PartitionsTest(ABC):
    '''Test of independence based on finite partitions'''
    def __init__(self, n_partitions_x, n_partitions_y):
        self.n_partitions_x = n_partitions_x
        self.n_partitions_y = n_partitions_y
        self.n_data = None
        self.dx = None
        self.dy = None
        self.statistic = None
            
    @abstractmethod
    def partition_statistic(self, emp_joint_dist, emp_marg_dist_x, emp_marg_dist_y):
        pass
    
    @abstractmethod
    def strongly_consistent_test(self):
        pass
    
    @abstractmethod
    def asymptotic_alpha_test(self, alpha):
        pass
    
    def equiprobable_partitions(self, samples, n_partitions):
        bounds = []
        for d in range(samples.shape[1]):
            dim_bounds_idx = np.linspace(0, samples.shape[0] - 1, n_partitions + 1).astype(int)
            dim_bounds = np.sort(samples[:,d])[dim_bounds_idx]
            dim_bounds[0] -= 1e-8
            bounds.append(dim_bounds)
        return bounds

    def fit(self, X, Y):
        self.n_data = X.shape[0]
        self.dx = X.shape[1]
        self.dy = Y.shape[1]
        bounds_x = self.equiprobable_partitions(X, self.n_partitions_x)
        bounds_y = self.equiprobable_partitions(Y, self.n_partitions_y)
        statistic = 0
        for i in range(self.n_partitions_x):
            for dx in range(X.shape[1]):
                partition_samples_x = (X[:,dx] > bounds_x[dx][i]) * (X[:,dx] <= bounds_x[dx][i+1])
                emp_marg_dist_x = partition_samples_x.sum() / self.n_data
                for j in range(self.n_partitions_y):
                    for dy in range(Y.shape[1]):
                        partition_samples_y = (Y[:,dy] > bounds_y[dy][j]) * (Y[:,dy] <= bounds_y[dy][j+1])
                        emp_marg_dist_y = partition_samples_y.sum() / self.n_data
                        emp_joint_dist = (partition_samples_x * partition_samples_y).sum() / self.n_data
                        statistic += self.partition_statistic(emp_joint_dist, emp_marg_dist_x, emp_marg_dist_y)
        self.statistic = statistic

    
class L1Test(PartitionsTest):
    '''Test of independence based on $L_1$ statistic'''
    def __init__(self, n_partitions_x, n_partitions_y):
        super().__init__(n_partitions_x, n_partitions_y)
            
    def partition_statistic(self, emp_joint_dist, emp_marg_dist_x, emp_marg_dist_y):
        return np.abs(emp_joint_dist - emp_marg_dist_x * emp_marg_dist_y)
            
    def strongly_consistent_test(self, C=1):
        c1 = np.sqrt(2 * np.log(2))
        threshold = (c1 * (np.sqrt(((self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) / self.n_data)))
        if self.statistic > C * threshold:
            return False
        else:
            return True
        
    def asymptotic_alpha_test(self, alpha):
        c2 = np.sqrt(2 / np.pi)
        sigma = np.sqrt(1 - 2 / np.pi)
        threshold = (c2 * np.sqrt(((self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) / self.n_data) + 
                     (sigma / np.sqrt(self.n_data) * norm.ppf(1 - alpha)))
        if self.statistic > threshold:
            return False
        else:
            return True
    
    
class LogLikelihoodTest(PartitionsTest):
    '''Test of independence based on log-likelihood statistic'''
    def __init__(self, n_partitions_x, n_partitions_y):
        super().__init__(n_partitions_x, n_partitions_y)
            
    def partition_statistic(self, emp_joint_dist, emp_marg_dist_x, emp_marg_dist_y):
        if emp_joint_dist > 0:
            return emp_joint_dist * np.log2(emp_joint_dist / (emp_marg_dist_x * emp_marg_dist_y))
        else:
            return 0
            
    def strongly_consistent_test(self, C=1):
        threshold = ((self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy) * 
                     (np.log(self.n_data + (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) + 1) / 
                     self.n_data)
        if self.statistic > C * threshold:
            return False
        else:
            return True
        
    def asymptotic_alpha_test(self, alpha):
        threshold = (norm.ppf(1 - alpha) * np.sqrt(2 * (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) + 
                    (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) / (2 * self.n_data)
        if self.statistic > threshold:
            return False
        else:
            return True
        
        
class PearsonChiSquareTest(PartitionsTest):
    '''Test of independence based on Pearson $\chi^2$ statistic'''
    def __init__(self, n_partitions_x, n_partitions_y):
        super().__init__(n_partitions_x, n_partitions_y)
            
    def partition_statistic(self, emp_joint_dist, emp_marg_dist_x, emp_marg_dist_y):
        return ((emp_joint_dist - emp_marg_dist_x * emp_marg_dist_y) ** 2 / 
               (emp_marg_dist_x * emp_marg_dist_y))
            
    def strongly_consistent_test(self, C=1):
        threshold = (2 * ((self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) ** (3 / 2) * 
                     (np.log(self.n_data + (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) + 1) / 
                    (self.n_data * np.log((self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)))) ** 2
        if self.statistic > C * threshold:
            return False
        else:
            return True
        
    def asymptotic_alpha_test(self, alpha):
        threshold = (norm.ppf(1 - alpha) * np.sqrt(2 * (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) + 
                    (self.n_partitions_x ** self.dx) * (self.n_partitions_y ** self.dy)) / self.n_data
        if self.statistic > threshold:
            return False
        else:
            return True