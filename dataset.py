import networkx as nx
import numpy as np
import torch as th
import os
import dgl

def collate(device):
    def collate_dev(samples):
        graphs, labels, seq_len = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        max_len = max(map(len, labels))
        batch_size = len(labels)
        pad_labels = th.zeros(batch_size, max_len)
        for i, l in enumerate(labels):
            pad_labels[i, :seq_len[i]] = th.tensor(labels[i])
        return batched_graph, pad_labels.to(device), th.tensor(seq_len).long().to(device)
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
        
        self.graphs = []
        self._gen_graph()
    
    def _get_content(self, files):
        all_content = []
        for f in files:
            cur_content = np.loadtxt(os.path.join(self.data_dir, f))
            all_content.append(cur_content)
        return all_content
    
    def _gen_graph(self):
        for f, e in zip(self.stroke_features, self.time_edges):
            g = dgl.DGLGraph()
            g.add_nodes(f.shape[0])
            g.ndata['x'] = th.Tensor(f)
            src, dst = map(list, zip(*e))
            g.add_edges(src, dst)
            self.graphs.append(g)
    
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
        (dgl.DGLGraph, numpy.ndarray, int)
            The graph, labels and actual length.
        """
        return self.graphs[idx], self.labels[idx], len(self.labels[idx])

if __name__ == "__main__":
    home = os.environ['HOME']
    data_dir = os.path.join(home, "data/IAMonDo/cls_5class")
    trainset = StrokeDataset(os.path.join(data_dir, "train"))
    
    train_loader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=collate)
    for it, (bg, label, seq_len) in enumerate(data_loader):
        print(label.size())
        print(seq_len)
        break
    