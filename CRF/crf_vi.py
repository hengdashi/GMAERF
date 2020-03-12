import numpy as np



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


class CRF_VI:

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


    def init_weights(self, seed=None):
        np.random.seed(seed)
        self.weights = np.random.uniform(size=(self.num_stats, self.num_classes))
        
        I = np.eye(self.num_classes)
        probs = np.zeros((self.num_vertices, self.num_classes))

        for i in range(self.num_vertices):

            probs[i] = I[self.Y[i]]

        self.probs = probs




    def fit(self, max_iter=10000, lr=1e-2, threshold=1e-6, reg=1e-3, print_every=1000):

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
            x_ = np.exp(self.Tx.dot(self.weights))
            Y_hat = self.Y_train
            Y_hat[self.ix_test] = x_[self.ix_test].argmax(axis=1)
            loss, dx_ = softmax_loss(x_, Y_hat)
            grad = self.Tx.T.dot(dx_)
            # V
            # print((self.A.sum(axis=1)==0).sum())
            # print(stats.min())
            
            # VxK

            # loss = ((probs_hat - self.probs)**2).mean()
            # self.weights *= (1-reg)
            # grad = np.zeros_like(self.weights)
            # for i in range(self.num_classes):
            #     grad.T[i] = (self.stats*((1.0 - probs_hat.T[i]) / z)[:,np.newaxis]*self.weights.T[i]).T.dot((probs_hat - self.probs).T[i])
                # print(grad.shape)
            
            self.weights -= (lr*grad + reg*self.weights)

            if it%print_every == print_every-1:
                print(f"Iteration {it+1:5d}, loss={loss:.8f}, accuracy={self.evaluate()*100:.2f}%")

    def evaluate(self):

        x_ = np.exp(self.Tx.dot(self.weights))
        Y_hat = self.Y_train
        Y_hat[self.ix_test] = x_[self.ix_test].argmax(axis=1)

        return (Y_hat==self.Y)[self.ix_test].mean()




if __name__ == '__main__':

    from load_data import *
    from statistics import *

    print(f"Using symmetric potentials:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, cora_features, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(nbr_count_sym_stat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")

    print(f"Using asymmetric potentials:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, cora_features, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(nbr_count_asym_stat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")

    hidden256_feature = np.loadtxt('hidden_emb256_gvae.content')
    hidden16_feature = np.loadtxt('hidden_emb16_gvae.content')

    print(f"Using symmetric potentials with 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(get_join_stat_function(nbr_count_sym_stat, feature_stat))
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using asymmetric potentials with 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(get_join_stat_function(nbr_count_asym_stat, feature_stat))
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()


    print(f"Using only 256 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden256_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(feature_stat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using symmetric potentials with 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(get_join_stat_function(nbr_count_sym_stat, feature_stat))
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()

    print(f"Using asymmetric potentials with 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(get_join_stat_function(nbr_count_asym_stat, feature_stat))
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()


    print(f"Using only 16 hidden embeddings:")
    cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, hidden16_feature, cora_klasses, cora_klasses_train, cora_ix_test)
    crf_vi.set_statistic_function(feature_stat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit(reg=0)
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")
    print()



    # print(f"Using binary factors:")
    # cora_klasses_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    # crf_vi = CRF_VI(cora_adj, cora_features, cora_klasses, cora_klasses_train, cora_ix_test)
    # crf_vi.set_statistic_function(binary_stat)
    # crf_vi.init_weights(seed=0)
    # crf_vi.fit(reg=0)
    # acc = crf_vi.evaluate()
    # print(f"Test accuracy: {acc*100:.2f}%")
    # print()
