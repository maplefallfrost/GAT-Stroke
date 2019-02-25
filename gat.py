import torch
import torch.nn as nn
import dgl.function as fn

class GraphAttention(nn.Module):
    def __init__(self,
             in_dim,
             edge_f_dim,
             out_dim,
             num_heads,
             feat_drop,
             attn_drop,
             alpha,
             residual=False):
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_e = nn.Sequential(
            nn.Linear(edge_f_dim, out_dim),
            self.leaky_relu,
            nn.Linear(out_dim, num_heads)
        )
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.residual = residual
        self.batch_norm = nn.BatchNorm1d(num_heads * out_dim)
        
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None
    
    def forward(self, g, inputs, last=False):
        # prepare
        h = self.feat_drop(inputs) # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1)) # NxHxD'
        head_ft = ft.transpose(0, 1) # HxNxD'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1) # NxHx1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1) # NxHx1
        g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        # 1. compute edge attention
        g.apply_edges(self.edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax(g)
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        # 3. apply normalizer
        ret = g.ndata['ft'] / g.ndata['z']
        # 4. residual:
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))
            else:
                resval = torch.unsqueeze(h, 1)
            ret = ret + resval
        if last == False:
            ret = self.batch_norm(ret.flatten(1))
        else:
            ret = ret.mean(1)
        return ret
   
    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        e_attn = self.attn_e(edges.data['x']).unsqueeze(-1)
        a = self.leaky_relu(e_attn + edges.src['a1'] + edges.dst['a2'])
        return {'a': a}
    
    def edge_softmax(self, g):
        # compute the max
        g.update_all(fn.copy_edge('a', 'a'), fn.max('a', 'a_max'))
        # minus the max and exp
        g.apply_edges(lambda edges: {'a': torch.exp(edges.data['a'] - edges.dst['a_max'])})
        # compute dropout
        g.apply_edges(lambda edges: {'a_drop': self.attn_drop(edges.data['a'])})
        # compute normalizer
        g.update_all(fn.copy_edge('a', 'a'), fn.sum('a', 'z'))
            
class GAT(nn.Module):
    def __init__(self,
             num_layers,
             in_dim,
             edge_f_dim,
             num_hidden,
             num_classes,
             heads,
             activation,
             feat_drop,
             attn_drop,
             alpha,
             residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            in_dim, edge_f_dim, num_hidden, heads[0], feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                num_hidden * heads[l-1], edge_f_dim, num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.gat_layers.append(GraphAttention(
            num_hidden * heads[-2], edge_f_dim, num_classes, heads[-1],
            feat_drop, attn_drop, alpha, residual))
    
    def forward(self, g):
        h = g.ndata['x']
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h, last=False).flatten(1)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](g, h, last=True)
        return logits