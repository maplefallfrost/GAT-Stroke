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
        batched_label_graphs.ndata['y'] = batched_label_graphs.ndata['y'].to(device)
        return batched_feature_graphs, batched_label_graphs
    return collate_dev

class StrokeDataset(object):
    """IAMonDo dataset class."""
    def __init__(self, data_dir):
        super(StrokeDataset, self).__init__()
        all_files = os.listdir(data_dir)
        
        self.data_dir = data_dir
        self.stroke_feature_files = [x for x in all_files if x.endswith('.stroke_feature')]
        self.label_files = [x for x in all_files if x.endswith('.label')]
        self.time_edge_files = [x for x in all_files if x.endswith('.time_edge')]
        key = lambda x: int(x.split(".")[0])
        self.stroke_feature_files.sort(key=key)
        self.label_files.sort(key=key)
        self.time_edge_files.sort(key=key)
        
        self.stroke_features = self._get_content(self.stroke_feature_files)
        self.labels = self._get_content(self.label_files)
        self.time_edges = self._get_content(self.time_edge_files)
        
        self.feature_graphs = []
        self.label_graphs = []
        self._gen_graph()
    
    def _get_content(self, files):
        all_content = []
        for f in files:
            cur_content = np.loadtxt(os.path.join(self.data_dir, f))
            all_content.append(cur_content)
        return all_content
    
    def _gen_graph(self):
        for f, l, e in zip(self.stroke_features, self.labels, self.time_edges):
            feature_graph = dgl.DGLGraph()
            feature_graph.add_nodes(f.shape[0])
            feature_graph.ndata['x'] = th.FloatTensor(f)
            src, dst = map(list, zip(*e))
            feature_graph.add_edges(src, dst)
            feature_graph.add_edges(feature_graph.nodes(), feature_graph.nodes())
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
    