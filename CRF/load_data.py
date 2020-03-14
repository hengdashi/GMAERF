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
	perm = np.random.permutation(n_total)
	ix_test = np.sort(perm[:n_test])
	ix_train = np.sort(perm[n_test:])


	return ix_train, ix_test


