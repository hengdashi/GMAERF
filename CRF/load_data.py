import numpy as np
import pandas as pd

cora_content = pd.read_csv('data/cora.content', delimiter='\t', header=None, index_col=0).sort_values(by=0)

cora_features = cora_content[range(1, 1434)].to_numpy(dtype=float)

cora_klasses = pd.factorize(cora_content[1434])[0]

cora_cites = np.vectorize(cora_content.index.get_loc)(np.loadtxt('data/cora.cites', dtype=int))

cora_adj = np.zeros((2708,2708))
cora_adj[cora_cites[:,1],cora_cites[:,0]] = 1



def train_test_split_node(adj, klasses, test_frac=0.2, seed=None):

	np.random.seed(seed)

	n_test = int(adj.shape[0]*test_frac)
	n_total = int(adj.shape[0])
	ix_test = np.random.permutation(n_total)[:n_test]
	# adj_train = adj.copy()
	# adj_train[ix_test] = 0
	# adj_train.T[ix_test] = 0
	klasses_train = klasses.copy()
	klasses_train[ix_test] = -1

	return klasses_train, ix_test


