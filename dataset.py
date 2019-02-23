import networkx as nx
import numpy as np
import torch as th
import os
import dgl

def collate(device):
    def collate_dev(samples):
        feature_graphs, label_graphs = map(list, zip(*samples))
        batched_feature_graphs = dgl.batch(feature_graphs)
        batched_label_graphs = dgl.batch(label_graphs)
        batched_feature_graphs.ndata['x'] = batched_feature_graphs.ndata['x'].to(device)
        batched_feature_graphs.edata['x'] = batched_feature_graphs.edata['x'].to(device)
        batched_label_graphs.ndata['y'] = batched_label_graphs.ndata['y'].to(device)
        return batched_feature_graphs, batched_label_graphs
    return collate_dev

class StrokeDataset(object):
    """IAMonDo dataset class."""
    def __init__(self, data_dir, edge_dir, set_type, num_classes):
        super(StrokeDataset, self).__init__()
        data_dir = os.path.join(data_dir, set_type)
        edge_dir = os.path.join(edge_dir, set_type)
        all_data_files = os.listdir(data_dir)
        all_edge_files = os.listdir(edge_dir)
        
        self.data_dir = data_dir
        self.edge_dir = edge_dir
        self.stroke_feature_files = [x for x in all_data_files if x.endswith('.stroke_feature')]
        self.binary_feature_files = [x for x in all_edge_files if x.endswith('.binary_feature')]
        self.label_files = [x for x in all_data_files if x.endswith('.label%d' % num_classes)]
        self.edge_files = [x for x in all_edge_files if x.endswith('.edge')]
        key = lambda x: int(x.split(".")[0])
        self.stroke_feature_files.sort(key=key)
        self.binary_feature_files.sort(key=key)
        self.label_files.sort(key=key)
        self.edge_files.sort(key=key)
        
        self.stroke_features = self._get_content(self.stroke_feature_files,
                                                 self.data_dir)
        self.binary_features = self._get_content(self.binary_feature_files,
                                                 self.edge_dir)
        self.labels = self._get_content(self.label_files,
                                        self.data_dir)
        self.edges = self._get_content(self.edge_files,
                                       self.edge_dir)
        
        self.feature_graphs = []
        self.label_graphs = []
        self._gen_graph()
    
    def _get_content(self, files, prefix):
        all_content = []
        for f in files:
            cur_content = np.loadtxt(os.path.join(prefix, f))
            all_content.append(cur_content)
        return all_content
    
    def _gen_graph(self):
        for f, b, l, e in zip(self.stroke_features, self.binary_features, self.labels, self.edges):
            feature_graph = dgl.DGLGraph()
            feature_graph.add_nodes(f.shape[0])
            feature_graph.ndata['x'] = th.FloatTensor(f)
            src, dst = map(list, zip(*e))
            feature_graph.add_edges(src, dst)
            feature_graph.add_edges(feature_graph.nodes(), feature_graph.nodes())
            feature_graph.set_e_initializer(dgl.init.zero_initializer)
            feature_graph.edges[src, dst].data['x'] = th.FloatTensor(b)
            self.feature_graphs.append(feature_graph)
            
            label_graph = dgl.DGLGraph()
            assert f.shape[0] == l.shape[0]
            label_graph.add_nodes(l.shape[0])
            label_graph.ndata['y'] = th.LongTensor(l)
            self.label_graphs.append(label_graph)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.label_files)
    
    def __getitem__(self, idx):
        """Get the i-th sample.
        
        Parameters
        ----------
        idx : int
           The sample index.
        Returns
        -------
        (dgl.DGLGraph, dgl.DGLGraph)
            The feature graph and label graph
        """
        return self.feature_graphs[idx], self.label_graphs[idx]
    