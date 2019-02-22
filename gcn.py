import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    
    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
    
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
    
    def forward(self, g, feature):
        g.ndata['h'] = feature
        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')
        
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class GCNNet(nn.Module):
    def __init__(self,
             in_feats,
             hidden_feats,
             num_classes):
        super(GCNNet, self).__init__()
        self.gcn1 = GCN(in_feats, hidden_feats, F.relu)
        self.gcn2 = GCN(hidden_feats, hidden_feats, F.relu)
        self.linear = nn.Linear(hidden_feats, num_classes)
    
    def forward(self, g):
        features = g.ndata['x']
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        logits = self.linear(x)
        return logits