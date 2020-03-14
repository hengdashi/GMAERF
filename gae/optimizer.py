import torch
import torch.nn.modules.loss
import torch.nn.functional as F

# TODO: the loss function might have some problem that causes
# the similarity score high in the air
# might be a problem of the sparse feature matrix
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight, target='adj'):
  cost = None
  if target == 'adj':
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
  elif target == 'feat-mlp' or target == 'feat-gcn':
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight, reduction='sum')
    # cost = norm * torch.sqrt(torch.sum(torch.sum(torch.pow(preds - labels, 2))))

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  if target == 'adj':
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
  elif target == 'feat-mlp' or target == 'feat-gcn':
    KLD = -0.5 * torch.sum(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
  # print(cost)
  # print(KLD)
  return cost + KLD
