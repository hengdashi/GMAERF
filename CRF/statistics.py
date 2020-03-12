import numpy as np
num_classes = 7


def nbr_count_sym_stat(A, X, Y):

    I = np.eye(num_classes)
    exp_Y = I[Y]
    # print(Y)

    stats = np.zeros((X.shape[0], num_classes))

    for i in range(X.shape[0]):

        stats[i, :] = I[Y[(A[i]+A.T[i])>0]].sum(axis=0)

    # print(stats.max())
    return stats



def nbr_count_asym_stat(A, X, Y):

    I = np.eye(num_classes)
    exp_Y = I[Y]
    # print(Y)

    stats = np.zeros((X.shape[0], 2*num_classes))

    for i in range(X.shape[0]):

        stats[i, :num_classes] = I[Y[(A[i])>0]].sum(axis=0)
        stats[i, num_classes:] = I[Y[(A.T[i])>0]].sum(axis=0)

    # print(stats.max())
    return stats

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
