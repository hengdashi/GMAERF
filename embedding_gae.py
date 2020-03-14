
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

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

args = {
  'dataset': 'cora',
  'epochs': 200,
  'h1_dim': 16,
  'h2_dim': 8,
  'lr': 1e-2,
  'weight_decay': 5e-4,
  # 'weight_decay': 0,
  'dropout': 0,
  'target': 'feat'
}


# In[4]:


# print(f"using {args['dataset']} dataset")

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
elif args['target'] == 'feat':
    pos_weight = torch.Tensor([float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()])
    norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)


# In[5]:


## training

model = GVAE(feat_dim, args['h1_dim'], args['h2_dim'], args['dropout'], target=args['target'])
optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

hidden_emb = None
for epoch in range(args['epochs']):
  t = time.time()
  model.train()
  optimizer.zero_grad()
  recovered, mu, logvar = model(features, adj_norm)
  if args['target'] == 'adj':
    labels = adj_label
  elif args['target'] == 'feat':
    labels = features
  loss = loss_function(preds=recovered, labels=labels,
                       mu=mu, logvar=logvar, n_nodes=n_nodes,
                       norm=norm, pos_weight=pos_weight,
                       target=args['target'])
  loss.backward()
  cur_loss = loss.item()
  optimizer.step()

  hidden_emb = mu.data.numpy()

  metric = 'cosine'

  if args['target'] == 'adj':
    roc_curr, ap_curr = gae.utils.get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
    sim_score = (paired_distances(recovered.detach().numpy(), labels.numpy(), metric=metric)).mean()
    preds = torch.gt(torch.sigmoid(recovered), 0.5).int()
    labels = labels.int()
    acc = torch.mean(torch.eq(preds, labels).float())
    tp = torch.nonzero(preds * labels).size(0)
    fp = torch.nonzero(preds * (labels - 1)).size(0)
    fn = torch.nonzero((preds - 1) * labels).size(0)
    tn = torch.nonzero((preds - 1) * (labels - 1)).size(0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"Epoch{(epoch+1):4}:", f"train_loss={cur_loss:.5f}",
          f"val_ap={ap_curr:.5f}", f"sim_score={sim_score:.5f}",
          f"time={(time.time()-t):.5f}", f"acc={acc:.5f}", f"tp={tp}", 
          f"fp={fp}", f"fn={fn}", f"tn={tn}", f"precision={precision:.5f}", 
          f"recall={recall:.5f}")
  elif args['target'] == 'feat':
    sim_score = (paired_distances(recovered.detach().numpy(), labels.numpy(), metric=metric)).mean()
    preds = torch.gt(torch.sigmoid(recovered), 0.5).int()
    labels = labels.int()
    acc = torch.mean(torch.eq(preds, labels).float())
    tp = torch.nonzero(preds * labels).size(0)
    fp = torch.nonzero(preds * (labels - 1)).size(0)
    fn = torch.nonzero((preds - 1) * labels).size(0)
    tn = torch.nonzero((preds - 1) * (labels - 1)).size(0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"Epoch{(epoch+1):4}:", f"train_loss={cur_loss:.5f}",
          f"sim_score={sim_score:.5f}", f"time={(time.time()-t):.5f}",
          f"acc={acc:.5f}", f"tp={tp}", f"fp={fp}", f"fn={fn}", f"tn={tn}",
          f"precision={precision:.5f}", f"recall={recall:.5f}")


# In[4]:


## validate

# roc_score, ap_score = gae.utils.get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))

papers = np.genfromtxt(f"data/cora.content", dtype=np.dtype(str))
# print(papers[:,0][:,np.newaxis])

# print(hidden_emb)
# print(papers[:,0][:,np.newaxis].astype(str))
# print(papers[:,-1][:,np.newaxis].astype(str))
X_train = hidden_emb
hidden_emb = torch.gt(torch.sigmoid(torch.from_numpy(hidden_emb.astype(float))), 0.5).int().numpy()
hidden_emb = np.append(papers[:,0][:,np.newaxis].astype(str), hidden_emb.astype(str), axis=1)
hidden_emb = np.append(hidden_emb.astype(str), papers[:,-1][:,np.newaxis].astype(str), axis=1)
print(hidden_emb)
y_train = papers[:,-1][:,np.newaxis].astype(str)

np.savetxt('hidden_emb_gvae.content', hidden_emb, fmt="%s")


# In[5]:


from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.preprocessing import LabelEncoder

classifier = SGDClassifier(verbose=1, max_iter=1000)
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)
print(sum(classifier.predict(X_train) == y_train) / y_train.shape[0])
