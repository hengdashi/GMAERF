import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F

from gae.model import GVAE
from gae.optimizer import loss_function
import gae.utils

from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import confusion_matrix

args = {
  'dataset': 'cora',
  'epochs': 200,
  'h1_dim': 32,
  'h2_dim': 16,
  'lr': 1e-2,
  'weight_decay': 3e-4,
  # 'weight_decay': 0,
  'dropout': 0,
  'target': 'adj'
}


print(f"using {args['dataset']} dataset")

# preprocessing
adj, features = gae.utils.load_data(args['dataset'])
n_nodes, feat_dim = features.shape
# print(f"adj dim: {adj.shape}")
# print(adj)
# print(f"fea dim: {features.shape}")
# print(features)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = gae.utils.mask_test_edges(adj)
adj = adj_train

adj_norm = gae.utils.preprocess_graph(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = torch.FloatTensor(adj_label.toarray())

if args['target'] == 'adj':
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
elif args['target'] == 'feat-mlp' or args['target'] == 'feat-gcn':
    # count(neg) / count(pos)
    pos_weight = torch.sqrt(torch.Tensor([float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()]))
    # norm = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2)
    norm = 1


## training

model = GVAE(feat_dim, args['h1_dim'], args['h2_dim'], args['dropout'], target=args['target'])
optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

hidden_emb = None
loss_values = []
for epoch in range(args['epochs']):
  t = time.time()
  model.train()
  optimizer.zero_grad()
  recovered, mu, logvar = model(features, adj_norm)
  if args['target'] == 'adj':
    labels = adj_label
  elif args['target'] == 'feat-mlp' or args['target'] == 'feat-gcn':
    labels = features
  loss = loss_function(preds=recovered, labels=labels,
                       mu=mu, logvar=logvar, n_nodes=n_nodes,
                       norm=norm, pos_weight=pos_weight,
                       target=args['target'])
  loss.backward()
  cur_loss = loss.item()
  optimizer.step()

  hidden_emb = mu.data.numpy()
  loss_values.append(cur_loss)

  metric = 'cosine'

  def confusion_mat(preds, labels):
    tp = torch.nonzero(preds * labels).size(0)
    fp = torch.nonzero(preds * (labels - 1)).size(0)
    fn = torch.nonzero((preds - 1) * labels).size(0)
    tn = torch.nonzero((preds - 1) * (labels - 1)).size(0)
    acc = torch.mean(torch.eq(preds, labels).float())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return acc, precision, recall, tp, fp, fn, tn

  if args['target'] == 'adj':
    roc_curr, ap_curr = gae.utils.get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
    sim_score = (paired_distances(recovered.detach().numpy(), labels.numpy(), metric=metric)).mean()
    preds = torch.gt(torch.sigmoid(recovered), 0.5).int()
    labels = labels.int()
    acc, precision, recall, tp, fp, fn, tn = confusion_mat(preds, labels)
    print(f"Epoch{(epoch+1):4}:", f"train_loss={cur_loss:.5f}",
          f"val_ap={ap_curr:.5f}", f"sim_score={sim_score:.5f}", f"precision={precision:.5f}", 
          f"recall={recall:.5f}", f"acc={acc:.5f}", f"tp={tp}", 
          f"fp={fp}", f"fn={fn}", f"tn={tn}",
          f"time={(time.time()-t):.5f}")
  elif args['target'] == 'feat-mlp' or args['target'] == 'feat-gcn':
    sim_score = (paired_distances(recovered.detach().numpy(), labels.numpy(), metric=metric)).mean()
    preds = torch.gt(torch.sigmoid(recovered), 0.5).int()
    labels = labels.int()
    acc, precision, recall, tp, fp, fn, tn = confusion_mat(preds, labels)
    print(f"Epoch{(epoch+1):4}:", f"train_loss={cur_loss:.5f}",
          f"sim_score={sim_score:.5f}",
          f"acc={acc:.5f}", f"precision={precision:.5f}", f"recall={recall:.5f}",
          f"tp={tp}", f"fp={fp}", f"fn={fn}", f"tn={tn}",
          f"time={(time.time()-t):.5f}")

# plt.plot(loss_values)
# plt.show()


## validate

roc_score, ap_score = gae.utils.get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

papers = np.genfromtxt(f"data/cora.content", dtype=np.dtype(str))

# print(papers[:,0][:,np.newaxis])
# print(hidden_emb)
# print(papers[:,0][:,np.newaxis].astype(str))
# print(papers[:,-1][:,np.newaxis].astype(str))
if args['target'] == 'adj' or args['target'] == 'feat-mlp':
  reconstructed = model.decode(torch.from_numpy(hidden_emb))
else:
  reconstructed = model.decode(torch.from_numpy(hidden_emb), adj_norm)
reconstructed = torch.gt(torch.sigmoid(reconstructed), 0.5).int().numpy()
reconstructed = np.append(papers[:,0][:,np.newaxis].astype(str), reconstructed.astype(str), axis=1)
reconstructed = np.append(reconstructed.astype(str), papers[:,-1][:,np.newaxis].astype(str), axis=1)

# np.savetxt('reconstruct.content', reconstructed, fmt="%s", delimiter='\t')



hidden_emb = torch.gt(torch.sigmoid(torch.from_numpy(hidden_emb.astype(float))), 0.5).int().numpy()
X_train = hidden_emb
hidden_emb = np.append(papers[:,0][:,np.newaxis].astype(str), hidden_emb.astype(str), axis=1)
hidden_emb = np.append(hidden_emb.astype(str), papers[:,-1][:,np.newaxis].astype(str), axis=1)
y_train = papers[:,-1].astype(str)

# np.savetxt('hidden_emb_gvae.content', hidden_emb, fmt="%s", delimiter='\t')


from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.preprocessing import LabelEncoder

# X_train = features
classifier = SGDClassifier()
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
# print(y_train)

classifier.fit(X_train, y_train)
print(classifier.score(X_train, y_train))


# import and setups

# from gcn.models import GCN
# import gcn.utils

# args = {
#   'dataset': 'cora',
#   'epochs': 200,
#   'hidden_dim': 16,
#   'lr': 1e-2,
#   'weight_decay': 5e-4,
#   'dropout': 0.5
# }


# Load data
# adj, features, labels, idx_train, idx_val, idx_test = gcn.utils.load_data()
# n_nodes, feat_dim = features.shape

# # Model and optimizer
# model = GCN(nfeat=feat_dim,
#             nhid=args['hidden_dim'],
#             nclass=labels.max().item() + 1,
#             dropout=args['dropout'])
# optimizer = optim.Adam(model.parameters(),
#                        lr=args['lr'],
#                        weight_decay=args['weight_decay'])


# training

# t_total = time.time()

# for epoch in range(args['epochs']):
#   t = time.time()
#   model.train()
#   optimizer.zero_grad()
#   output = model(features, adj)
#   loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#   acc_train = gcn.utils.accuracy(output[idx_train], labels[idx_train])
#   loss_train.backward()
#   optimizer.step()

#   loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#   acc_val = gcn.utils.accuracy(output[idx_val], labels[idx_val])
#   print(f'Epoch: {(epoch+1):04d}',
#         f'loss_train: {loss_train.item():.4f}',
#         f'acc_train: {acc_train.item():.4f}',
#         f'loss_val: {loss_val.item():.4f}',
#         f'acc_val: {acc_val.item():.4f}',
#         f'time: {(time.time() - t):.4f}s')

# npemb = model.hidden_emb.detach().numpy()
# print(npemb.shape)
# np.savetxt('hidden_emb.content', npemb)

# print("Optimization Finished!")
# print(f"Total time elapsed: {time.time() - t_total:.4f}s")


# testing

# model.eval()
# output = model(features, adj)
# loss_test = F.nll_loss(output[idx_test], labels[idx_test])
# acc_test = gcn.utils.accuracy(output[idx_test], labels[idx_test])
# print(f"Test set results:",
#       f"loss= {loss_test.item():.4f}",
#       f"accuracy= {acc_test.item():.4f}")
