import numpy as np

from load_data import *


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(1e-8+probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


class CRF_Gibbs:
    def __init__(self, A, X, Y, Y_train, ix_test):

        self.A = A
        self.X = X
        self.Y = Y
        self.Y_train = Y_train
        self.ix_test = ix_test

        self.num_classes = Y.max()+1
        self.num_vertices = A.shape[0]
        self.num_edges = A.sum()


    def set_statistic_function(self, func):
        self.statistic_function = func
        Y_init = self.Y_train.copy()
        Y_init[self.ix_test] = 0
        self.Tx = func(self.A, self.X, Y_init)
        # self.Tx = Tx
        self.num_factors, self.num_stats = self.Tx.shape

    def sample(self, n_iter=100):
        n_unknown = self.ix_test.shape[0]
        Y_hat = self.Y_train.copy()
        Y_hat[self.ix_test] = np.random.choice(self.num_classes, size=n_unknown)

        t_no_change = 0

        for it in range(n_iter):
            u = self.ix_test[it%n_unknown]
            logdist = (self.Tx[u,:].dot(self.weights))
            logdist -= logdist.min()
            dist = np.exp(logdist)
            dist /= dist.sum()
            # new_val = logdist.argmax()
            new_val = np.random.choice(self.num_classes, p=dist)

            if new_val == Y_hat[u]:
                t_no_change += 1
            else:
                t_no_change = 0

            Y_hat[u] = new_val
            # print(u, new_val, dist, self.Tx[u], self.weights)
            # exit()
            self.Tx[u,:] = self.statistic_function(self.A, self.X[[u],:], Y_hat)

            if t_no_change == 3:
                # print(it)
                break

        return Y_hat

    def map(self):
        n_unknown = self.ix_test.shape[0]
        Y_hat = self.Y_train.copy()

        for u in self.ix_test:
            logdist = (self.Tx[u,:].dot(self.weights))
            Y_hat[u] = logdist.argmax()

        return Y_hat


    def init_weights(self, seed=None):
        np.random.seed(seed)
        Y_hat = self.Y_train.copy()
        Y_hat[self.ix_test] = np.random.choice(self.num_classes, size=self.ix_test.shape[0])
        self.Tx = self.statistic_function(self.A, self.X, Y_hat)
        self.num_stats = self.Tx.shape[1]
        self.weights = np.random.uniform(size=(self.num_stats, self.num_classes))


    def fit(self, max_iter=10000, lr=1e-3, threshold=1e-6, reg=1e-3, n_samples=1, print_every=1000):

        # for it in range(max_iter):

        #     # VxK
        #     probs_hat_ = self.stats.dot(self.weights**2)
        #     assert 0<=probs_hat.min()<=
        #     # V
        #     z = probs_hat_.sum(axis=1)
        #     # print((self.A.sum(axis=1)==0).sum())
        #     # print(stats.min())
            
        #     # VxK
        #     probs_hat = probs_hat_/z[:,np.newaxis]

        #     loss = ((probs_hat - self.probs)**2).mean()
        #     self.weights *= (1-reg)
        #     grad = np.zeros_like(self.weights)
        #     for i in range(self.num_classes):
        #         grad.T[i] = (self.stats*((1.0 - probs_hat.T[i]) / z)[:,np.newaxis]*self.weights.T[i]).T.dot((probs_hat - self.probs).T[i])
        #         # print(grad.shape)
            
        #     self.weights -= (lr*grad + reg*self.weights)

        #     if it%100 == 99:
        #         print(f"Iteration {it+1:5d}, loss={loss:.8f}, accuracy={self.evaluate()*100:.2f}%")

        for it in range(max_iter):

            # VxK
            gradsum = np.zeros_like(self.weights)

            # for i in range(n_samples):
                # Y_hat = self.sample()
            Y_hat = self.map()
            x_ = np.exp(self.Tx.dot(self.weights))
            loss, dx_ = softmax_loss(x_, Y_hat)
            gradsum += self.Tx.T.dot(dx_)
            
            self.weights -= (lr*(gradsum/n_samples) + reg*self.weights)

            if it%print_every == print_every-1:
                print(f"Iteration {it+1:5d}, loss={loss:.8f}, accuracy={self.evaluate()*100:.2f}%")

    def evaluate(self):

        Y_hat = self.map()

        return (Y_hat==self.Y)[self.ix_test].mean()




if __name__ == '__main__':

    from load_data import *
    from statistics import *

    print(f"Using symmetric potentials:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, cora_features, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(nbr_count_sym_stat)
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit()
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using asymmetric potentials:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, cora_features, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(nbr_count_asym_stat)
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit()
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    hidden256_feature = np.loadtxt('hidden_emb256_gvae.content')
    hidden16_feature = np.loadtxt('hidden_emb16_gvae.content')

    print(f"Using symmetric potentials with 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(get_join_stat_function(nbr_count_sym_stat, feature_stat))
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using asymmetric potentials with 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(get_join_stat_function(nbr_count_asym_stat, feature_stat))
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using only 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(feature_stat)
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using symmetric potentials with 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(get_join_stat_function(nbr_count_sym_stat, feature_stat))
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using asymmetric potentials with 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(get_join_stat_function(nbr_count_asym_stat, feature_stat))
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using only 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_gibbs = CRF_Gibbs(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_gibbs.set_statistic_function(feature_stat)
    crf_gibbs.init_weights(seed=0)
    crf_gibbs.fit(reg=0)
    acc = crf_gibbs.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()
    