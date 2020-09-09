import torch
import torch.nn as nn
import dgl.function as fn

class EGATLayer(nn.Module):
    def __init__(self,
             in_dim,
             edge_f_dim,
             out_dim,
             edge_out_dim,
             num_heads,
             feat_drop,
             attn_drop,
             alpha,
             temperature,
             edge_feature_attn,
             edge_update):

        super(EGATLayer, self).__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.edge_feature_attn = edge_feature_attn
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
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
        self.node_batch_norm = nn.BatchNorm1d(num_heads * out_dim)
        
        self.edge_update = edge_update
        if self.edge_update:
            self.edge_update_node_fc = nn.Linear(2 * num_heads * out_dim, edge_out_dim, bias=False)
            self.edge_update_edge_fc = nn.Linear(edge_f_dim, edge_out_dim, bias=False)
            self.edge_update_final_fc = nn.Linear(2 * edge_out_dim, edge_out_dim, bias=False)
            self.edge_batch_norm = nn.BatchNorm1d(edge_out_dim)

    def forward(self, node_graph, node_inputs, edge_inputs):
        # update node representation
        h = self.feat_drop(node_inputs) # NxD
        edge_h = self.feat_drop(edge_inputs)
        node_graph.edata.update({'h': edge_h})

        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1)) # NxHxD'
        head_ft = ft.transpose(0, 1) # HxNxD'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1) # NxHx1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1) # NxHx1
        node_graph.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        # 1. compute edge attention
        node_edge_attention = lambda edges: self.edge_attention(edges, self.edge_feature_attn)
        node_graph.apply_edges(node_edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax(node_graph)
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        node_graph.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        # 3. apply normalizer
        node_outputs = node_graph.ndata['ft'] / node_graph.ndata['z']
        node_outputs = node_outputs.flatten(1)
        # 4. residual:
        if node_inputs.size(-1) == node_outputs.size(-1):
            node_outputs += node_inputs
        # 5. batch norm:
        node_outputs = self.leaky_relu(node_outputs)
        node_outputs = self.node_batch_norm(node_outputs)

        if self.edge_update:
            node_graph.ndata.update({'h': node_outputs})
            node_graph.apply_edges(self.edge_update_fn)
            edge_outputs = node_graph.edata['h']
            if edge_inputs.size(-1) == edge_outputs.size(-1):
                edge_outputs += edge_inputs
            edge_outputs = self.leaky_relu(edge_outputs)
            edge_outputs = self.edge_batch_norm(edge_outputs)

        return node_outputs, edge_outputs
    
    def edge_attention(self, edges, edge_feature_attn):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = edges.src['a1'] + edges.dst['a2']
        if edge_feature_attn:
            e_attn = self.attn_e(edges.data['h']).unsqueeze(-1)
            a = a + e_attn
        a = self.leaky_relu(a)
        a = a * self.temperature
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
    
    def edge_update_fn(self, edges):
        node_f = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        node_f = self.edge_update_node_fc(node_f)
        node_f = self.leaky_relu(node_f)

        edge_f = self.edge_update_edge_fc(edges.data['h'])
        edge_f = self.leaky_relu(edge_f)

        final_h = torch.cat([node_f, edge_f], dim=-1)
        final_h = self.edge_update_final_fc(final_h)

        return {'h': final_h}
    
            
class EGAT(nn.Module):
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
             temperature,
             edge_feature_attn,
             edge_update):

        super(EGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(EGATLayer(
            in_dim, edge_f_dim, num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, alpha, temperature, edge_feature_attn, edge_update))
        # hidden layers
        for k in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(EGATLayer(
                num_hidden * heads[k-1], num_hidden, num_hidden, num_hidden, heads[k],
                feat_drop, attn_drop, alpha, temperature, edge_feature_attn, edge_update))
        
        self.node_project = nn.Sequential(
            nn.Linear(num_hidden * heads[-1], num_hidden * heads[-1]),
            self.activation,
            nn.Linear(num_hidden * heads[-1], num_classes))
    
    def forward(self, g):
        node_h = g.ndata['x']
        edge_h = g.edata['x']
        for k in range(self.num_layers):
            node_h, edge_h = self.gat_layers[k](g, node_h, edge_h)
        # output projection
        logits = self.node_project(node_h)
        return logits
