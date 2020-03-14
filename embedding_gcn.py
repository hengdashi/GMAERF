
from gcn.models import GCN
import gcn.utils
from torch import optim
import time
import torch.nn.functional as F
import numpy as np

for n_hidden in [4,8,16,32,64,128,256]:
# 79.70 83.00 83.70 83.70 83.70 82.70
  args = {
    'dataset': 'cora',
    'epochs': 1000,
    'hidden_dim': n_hidden,
    'lr': 1e-2,
    'weight_decay': 5e-4,
    'dropout': 0.5
  }

  adj, features, labels, idx_train, idx_val, idx_test = gcn.utils.load_data()
  n_nodes, feat_dim = features.shape

  # Model and optimizer
  model = GCN(nfeat=feat_dim,
              nhid=args['hidden_dim'],
              nclass=labels.max().item() + 1,
              dropout=args['dropout'])
  optimizer = optim.Adam(model.parameters(),
                         lr=args['lr'],
                         weight_decay=args['weight_decay'])


  t_total = time.time()

  for epoch in range(args['epochs']):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = gcn.utils.accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = gcn.utils.accuracy(output[idx_val], labels[idx_val])
    
    # print(f'Epoch: {(epoch+1):04d}',
          # f'loss_train: {loss_train.item():.4f}',
          # f'acc_train: {acc_train.item():.4f}',
          # f'loss_val: {loss_val.item():.4f}',
          # f'acc_val: {acc_val.item():.4f}',
          # f'time: {(time.time() - t):.4f}s')

  npemb = model.hidden_emb.detach().numpy()
  print(npemb.shape)
  np.savetxt(f'hidden_emb_{n_hidden}.content', npemb)

  print("Optimization Finished!")
  print(f"Total time elapsed: {time.time() - t_total:.4f}s")

  model.eval()
  output = model(features, adj)
  loss_test = F.nll_loss(output[idx_test], labels[idx_test])
  acc_test = gcn.utils.accuracy(output[idx_test], labels[idx_test])
  print(f"Test set results:",
        f"loss= {loss_test.item():.4f}",
        f"accuracy= {acc_test.item():.4f}")




