import numpy as np
num_classes = 7
I = np.eye(num_classes)


class SufficientStatistic:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.out_nbrs = [None]*self.A.shape[0]
        self.in_nbrs = [None]*self.A.shape[0]
        self.all_nbrs = [None]*self.A.shape[0]


        for i in range(self.A.shape[0]):
            self.out_nbrs[i] = list(set(np.where(self.A[:,i])[0]))
            self.in_nbrs[i] = list(set(np.where(self.A[i,:])[0]))
            self.all_nbrs[i] = list(set(np.where(self.A[i,:]+self.A[:,i])[0]))



class NbrInfoSymmetricStat(SufficientStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.X is not None:
            self.stats = np.zeros((self.A.shape[0], num_classes+self.X.shape[1]))
            self.stats[:, num_classes:] = self.X
        else:
            self.stats = np.zeros((self.A.shape[0], num_classes))

        for nbr in range(self.A.shape[0]):
            self.stats[nbr, :num_classes] = I[self.Y[self.all_nbrs[nbr]]].sum(axis=0)

    def update_node(self, node, val):

        
        self.Y[node] = val
        exp_Y = I[self.Y]
        self.stats[node, :num_classes] = I[self.Y[self.all_nbrs[node]]].sum(axis=0)

        for nbr in self.all_nbrs[node]:
            self.stats[nbr, :num_classes] = I[self.Y[self.all_nbrs[node]]].sum(axis=0)

    def update_all(self, Y_new):
        self.Y = Y_new

        for node in range(Y_new.shape[0]):
            self.stats[node, :num_classes] = I[self.Y[self.all_nbrs[node]]].sum(axis=0)

class NbrInfoAsymmetricStat(SufficientStatistic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.X is not None:
            self.stats = np.zeros((self.A.shape[0], 2*num_classes+self.X.shape[1]))
            self.stats[:, 2*num_classes:] = self.X
        else:
            self.stats = np.zeros((self.A.shape[0], 2*num_classes))

        for nbr in range(self.A.shape[0]):
            self.stats[nbr, :num_classes] = I[self.Y[self.out_nbrs[nbr]]].sum(axis=0)
            self.stats[nbr, num_classes:2*num_classes] = I[self.Y[self.in_nbrs[nbr]]].sum(axis=0)

    def update_node(self, node, val):

        
        self.Y[node] = val

        self.stats[node, :num_classes] = I[self.Y[self.out_nbrs[node]]].sum(axis=0)
        self.stats[node, num_classes:2*num_classes] = I[self.Y[self.in_nbrs[node]]].sum(axis=0)

        for nbr in self.out_nbrs[node]:
            self.stats[nbr, num_classes:2*num_classes] = I[self.Y[self.in_nbrs[nbr]]].sum(axis=0)
        
        for nbr in self.in_nbrs[node]:
            self.stats[nbr, :num_classes] = I[self.Y[self.out_nbrs[nbr]]].sum(axis=0)

    def update_all(self, Y_new):
        self.Y = Y_new

        for node in range(Y_new.shape[0]):
            self.stats[node, num_classes:2*num_classes] = I[self.Y[self.in_nbrs[node]]].sum(axis=0)
            self.stats[node, :num_classes] = I[self.Y[self.out_nbrs[node]]].sum(axis=0)

# def nbr_count_sym_stat(A, X, Y):

#     exp_Y = I[Y]
#     # print(Y)

#     stats = np.zeros((X.shape[0], num_classes))

#     for i in range(X.shape[0]):

#         stats[i, :] = I[Y[(A[i]+A.T[i])>0]].sum(axis=0)

#     # print(stats.max())
#     return stats



# def nbr_count_asym_stat(A, X, Y):

#     I = np.eye(num_classes)
#     exp_Y = I[Y]
#     # print(Y)

#     stats = np.zeros((X.shape[0], 2*num_classes))

#     for i in range(X.shape[0]):

#         stats[i, :num_classes] = I[Y[(A[i])>0]].sum(axis=0)
#         stats[i, num_classes:] = I[Y[(A.T[i])>0]].sum(axis=0)

#     # print(stats.max())
#     return stats

def feature_stat(A,X,Y):
    return X


def get_join_stat_function(*funcs):

    def join_stat(A, X, Y):

        all_stats = [func(A,X,Y) for func in funcs]

        return np.hstack(all_stats)

    return join_stat

from scipy.spatial.distance import cosine

def binary_stat(A, X, Y):

    E = int(A.sum())
    I = np.eye(num_classes)

    stats = np.zeros((E, num_classes**2+2*X.shape[1]))
    print(X.shape)
    print(Y.shape)
    print(stats.shape)
    for i, u,v in list(zip(np.arange(E), *np.where(A))):
        klass_feature = I[Y[u]].T.dot(I[Y[v]]).flatten()
        print(klass_feature.shape)
        # print(I[Y[u,:]].dot(I[Y[v,:]].T).shape)
        # print(c.shape)
        # print(klass_feature.shape)
        stats[i,:] = np.concatenate([klass_feature, X[u,:], X[v,:]])

    return stats
