import torch
import sys
import itertools
from torch.autograd import Variable

def batchify(batch_list, gpu, volatile_flag=False):
    """
    Input: 
        batch_list: list of raw features and labels, [[features, labels], ...]
            features: nested list, e.g. [[[0.1,0.2,0.3],[0.2,0.3,0.1]]]
            labels: tag index list, e.g. [1, 1, 1]
        gpu: boolean. indicates whether to use gpu.
        volatile_flag: boolean.
    Output:
        a dict containing zero padding for raw features, the keys are:
        'feature_tensor': Variable, [batch_size*max_stroke_len, max_point_len, feature_dim]
        'label_tensor': Variable, [batch_size, max_stroke_len]
        'stroke_seq_lengths': Tensor, [batch_size]
        'point_seq_lengths': Tensor, [batch_size*max_stroke_len]
        'point_seq_recover': Tensor, [batch_size*max_stroke_len]
    """
    if not batch_list:
        print("batch_list is empty!")
        sys.exit(-1)
        
    batch_size = len(batch_list)
    features = [sample[0] for sample in batch_list]
    feature_dim = len(features[0][0][0])
    labels = [sample[1] for sample in batch_list]
    
    stroke_seq_lengths = torch.LongTensor(list(map(len, features)))
    max_seq_len = int(stroke_seq_lengths.max())
    label_tensor = Variable(torch.zeros(batch_size, max_seq_len), volatile=volatile_flag).long()
    
    # pad label
    for idx, (label, seq_len) in enumerate(zip(labels, stroke_seq_lengths)):
        label_tensor[idx, :seq_len] = torch.LongTensor(label) + 1
    
    # pad feature
    point_length_list = [[len(stroke) for stroke in doc] for doc in features]
    assert len(point_length_list[0]) == len(labels[0])
    
    max_point_len = max(itertools.chain(*point_length_list))
    feature_tensor = Variable(torch.zeros(batch_size, max_seq_len, max_point_len, feature_dim), volatile=volatile_flag)
    point_seq_lengths = torch.zeros(batch_size, max_seq_len).long()    
    
    for i, doc in enumerate(features):
        for j, stroke in enumerate(doc):
            point_seq_lengths[i, j] = point_length_list[i][j]
            for k, point in enumerate(stroke):
                feature_tensor[i, j, k, :] = torch.FloatTensor(point)
                
    feature_tensor = feature_tensor.view(batch_size*max_seq_len, max_point_len, -1)
    point_seq_lengths = point_seq_lengths.view(batch_size*max_seq_len,)
    point_seq_lengths, point_perm_idx = point_seq_lengths.sort(0, descending=True)
    feature_tensor = feature_tensor[point_perm_idx]
    
    _, point_seq_recover = point_perm_idx.sort(0, descending=False)
    
    if gpu:
        feature_tensor = feature_tensor.cuda()
        label_tensor = label_tensor.cuda()
        stroke_seq_lengths = stroke_seq_lengths.cuda()
        point_seq_lengths = point_seq_lengths.cuda()
        
    return {'feature_tensor': feature_tensor,
            'label_tensor': label_tensor,
            'stroke_seq_lengths': stroke_seq_lengths,
            'point_seq_lengths': point_seq_lengths,
            'point_seq_recover': point_seq_recover,
           }
        
        