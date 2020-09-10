import numpy as np
import torch as th
import os
import dgl

def to_device(batch, device):
    fg, lg = batch
    fg.ndata['x'] = fg.ndata['x'].to(device)
    fg.edata['x'] = fg.edata['x'].to(device)
    lg.ndata['y'] = lg.ndata['y'].to(device)
    return fg, lg


def collate(samples):
    feature_graphs, label_graphs = map(list, zip(*samples))
    batched_feature_graphs = dgl.batch(feature_graphs)
    batched_label_graphs = dgl.batch(label_graphs)
    return batched_feature_graphs, batched_label_graphs


class StrokeDataset(object):
    """IAMonDo dataset class."""
    def __init__(self, data_dir, edge_dir, set_type, num_classes, cache_data=True):
        """
        If the whole data can be stored in memory, set cache_data=True.
        Otherwise, change the data format from plain text to other format(like .npy) to speed up reading data from disk.
        """
        data_dir = os.path.join(data_dir, set_type)
        edge_dir = os.path.join(edge_dir, set_type)
        all_data_files = os.listdir(data_dir)
        all_edge_files = os.listdir(edge_dir)
        
        self.data_dir = data_dir
        self.edge_dir = edge_dir
        self.cache_data = cache_data
        self.stroke_feature_files = [x for x in all_data_files if x.endswith('.stroke_feature')]
        self.binary_feature_files = [x for x in all_edge_files if x.endswith('.binary_feature')]
        self.label_files = [x for x in all_data_files if x.endswith('.label%d' % num_classes)]
        self.edge_files = [x for x in all_edge_files if x.endswith('.edge')]

        key = lambda x: int(x.split(".")[0])
        self.stroke_feature_files.sort(key=key)
        self.binary_feature_files.sort(key=key)
        self.label_files.sort(key=key)
        self.edge_files.sort(key=key)

        self.feature_graphs = dict()
        self.label_graphs = dict()

    def _get_content(self, file_name, prefix):
        cur_content = np.loadtxt(os.path.join(prefix, file_name))
        return cur_content
    
    def _get_graph(self, stroke_feature, binary_feature, label, edge):
        feature_graph = dgl.DGLGraph()
        feature_graph.add_nodes(stroke_feature.shape[0])
        feature_graph.ndata['x'] = th.FloatTensor(stroke_feature)
        src, dst = map(list, zip(*edge))
        feature_graph.add_edges(src, dst)
        feature_graph.add_edges(feature_graph.nodes(), feature_graph.nodes())
        feature_graph.edges[src, dst].data['x'] = th.FloatTensor(binary_feature)
        
        label_graph = dgl.DGLGraph()
        assert stroke_feature.shape[0] == label.shape[0]
        label_graph.add_nodes(label.shape[0])
        label_graph.ndata['y'] = th.LongTensor(label)

        return feature_graph, label_graph
    
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
        if idx not in self.feature_graphs:
            stroke_feature = self._get_content(self.stroke_feature_files[idx], self.data_dir)
            binary_feature = self._get_content(self.binary_feature_files[idx], self.edge_dir)
            label = self._get_content(self.label_files[idx], self.data_dir)
            edge = self._get_content(self.edge_files[idx], self.edge_dir)

            feature_graph, label_graph = self._get_graph(stroke_feature, binary_feature, label, edge)
            if self.cache_data:
                self.feature_graphs[idx] = feature_graph
                self.label_graphs[idx] = label_graph
        else:
            feature_graph, label_graph = self.feature_graphs[idx], self.label_graphs[idx]

        return feature_graph, label_graph
    