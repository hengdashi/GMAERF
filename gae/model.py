import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class GVAE(nn.Module):
  def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, target='adj'):
    super(GVAE, self).__init__()
    self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
    self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
    self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
    if target == 'adj':
      self.dc = InnerProductDecoder(dropout, act=lambda x: x)
    elif target == 'feat':
      # self.dc = MLPDecoder(dropout)
      self.dc = GCNDecoder(hidden_dim2, hidden_dim1, input_feat_dim, dropout, act=F.relu)

  def encode(self, x, adj):
    hidden1 = self.gc1(x, adj)
    return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def forward(self, x, adj):
    mu, logvar = self.encode(x, adj)
    z = self.reparameterize(mu, logvar)
    return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
  """Decoder for using inner product for prediction."""

  def __init__(self, dropout, act=torch.sigmoid):
    super(InnerProductDecoder, self).__init__()
    self.dropout = dropout
    self.act = act

  def forward(self, z):
    z = F.dropout(z, self.dropout, training=self.training)
    adj = self.act(torch.mm(z, z.t()))
    return adj

class MLPDecoder(nn.Module):
  """MLP decoder for prediction"""

  def __init__(self, dropout, act=F.relu):
    super(MLPDecoder, self).__init__()
    self.dropout = dropout
    self.act = act
    self.fc1 = nn.Linear(256, 512)
    self.fc2 = nn.Linear(512, 1433)

  def forward(self, z):
    z = F.dropout(z, self.dropout, training=self.training)
    return self.fc2(self.act(self.fc1(z)))

class GCNDecoder(nn.Module):
  """MLP decoder for prediction"""

  def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, act=F.relu):
    super(GCNDecoder, self).__init__()
    self.dropout = dropout
    self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=act)
    self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

  def forward(self, z, adj):
    z = self.gc1(z, adj)
    z = F.dropout(z, self.dropout, training=self.training)
    return self.gc2(z, adj)
