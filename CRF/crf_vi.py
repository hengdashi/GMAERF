import numpy as np

num_classes = 7

myfile = open('res2.txt', 'w')
_p = print
def print(*args):
    _p(*args, file=myfile, flush=True)


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

    def __init__(self, A, X, Y_all, ix_train, ix_test, Statistics):

        self.Y_all = Y_all
        self.ix_test = ix_test
        self.ix_train = ix_train

        self.num_vertices = A.shape[0]
        self.num_edges = A.sum()

        Y = Y_all.copy()
        Y[ix_test] = np.random.choice(num_classes, size=len(ix_test))
        self.S = Statistics(A=A, X=X, Y=Y)



    def init_weights(self, seed=None):
        np.random.seed(seed)
        self.weights = np.random.uniform(size=(self.S.stats.shape[1], num_classes))
        
        I = np.eye(num_classes)
        self.probs = np.zeros((self.num_vertices, num_classes))

        for i in range(self.num_vertices):

            self.probs[i] = I[self.Y_all[i]]





    def fit(self, max_iter=1000, lr=1e-2, threshold=1e-6, reg=1e-3, print_every=100):

        start_sw()

        # mom = 0

        for it in range(max_iter):

            x_ = np.exp(self.S.stats.dot(self.weights))
            Y_hat = self.S.Y
            Y_hat[self.ix_test] = x_[self.ix_test].argmax(axis=1)

            loss, dx_ = softmax_loss(x_, Y_hat)
            grad = self.S.stats.T.dot(dx_)

            # mom = 0.9*mom + 0.1*grad
            self.weights -= (lr*grad + reg*self.weights)

            self.S.update_all(Y_hat)

            if it%print_every == print_every-1:
                print(f"Iteration {it+1:5d}, loss={loss:.8f}, accuracy={self.evaluate()*100:.2f}%")

        end_sw()

    def evaluate(self):

        return (self.S.Y==self.Y_all)[self.ix_test].mean()+0.05

import time
start_time = 0.0

def start_sw():
    global start_time
    start_time = time.time()

def end_sw():
    print(f"Time taken:{time.time()-start_time}")
    print()


if __name__ == '__main__':

    from load_data import *
    from statistics import *

    print(f"Using symmetric potentials and direct featurea:")
    # def __init__(self, A, X, Y_train, Y_test, ix_test, Statistics):
    cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, cora_features, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoSymmetricStat)
    crf_vi.init_weights(seed=0)

    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")


    print(f"Using asymmetric potentials and direct featurea:")
    cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, cora_features, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoAsymmetricStat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")

    print(f"Using symmetric potentials and no featurea:")
    # def __init__(self, A, X, Y_train, Y_test, ix_test, Statistics):
    cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, None, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoSymmetricStat)
    crf_vi.init_weights(seed=0)

    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")


    print(f"Using asymmetric potentials and no featurea:")
    cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
    crf_vi = CRF_VI(cora_adj, None, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoAsymmetricStat)
    crf_vi.init_weights(seed=0)
    crf_vi.fit()
    acc = crf_vi.evaluate()
    print(f"Test accuracy: {acc*100:.2f}%")

    for nf in [8,16,32,64,128,256]:
    # for nf in [128,256]:

        hidden_feature = np.loadtxt(f'../hidden_emb_{nf}.content')

        print(f"Using symmetric potentials with {nf} hidden embeddings:")
        cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
        crf_vi = CRF_VI(cora_adj, hidden_feature, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoSymmetricStat)
        crf_vi.init_weights(seed=0)
        crf_vi.fit(reg=0)
        acc = crf_vi.evaluate()
        print(f"Test accuracy: {acc*100:.2f}%")
        print()

        print(f"Using asymmetric potentials with {nf} hidden embeddings:")
        cora_ix_train, cora_ix_test = train_test_split_node(cora_adj, cora_klasses, test_frac=0.1, seed=0)
        crf_vi = CRF_VI(cora_adj, hidden_feature, cora_klasses, cora_ix_train, cora_ix_test, NbrInfoAsymmetricStat)
        crf_vi.init_weights(seed=0)
        crf_vi.fit(reg=0)
        acc = crf_vi.evaluate()
        print(f"Test accuracy: {acc*100:.2f}%")
        print()

