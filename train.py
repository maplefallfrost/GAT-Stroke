import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from utils import batchify
from tqdm import tqdm_notebook
from torch.nn.utils import clip_grad_norm

def neg_log_likelihood_loss(score, label, average_batch):
    batch_size, seq_len = label.size(0), label.size(1)
    score = score.view(batch_size*seq_len, -1)
    label = label.view(batch_size*seq_len)
    score = F.log_softmax(score, 1)
    nll_loss = nn.NLLLoss(ignore_index=0, size_average=False)
    total_loss = nll_loss(score, label)
    if average_batch:
        total_loss /= batch_size
    return total_loss

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(data, model, optimizer, loss_func, config):
    train_batch_list, valid_batch_list, test_batch_list = data
    num_train_sample = len(train_batch_list)
    batch_size = config.batch_size
    
    model_name = model.__class__.__name__
    optim_name = optimizer.__class__.__name__
    model_load_dir = os.path.join(config.model_dir, "pretrained", model_name)
    model_save_dir = os.path.join(config.model_dir, model_name)
    optim_load_dir = os.path.join(config.model_dir, "pretrained", optim_name)
    optim_save_dir = os.path.join(config.model_dir, optim_name)

    if os.path.exists(model_load_dir) and not config.train_from_scratch:
        model.load_state_dict(torch.load(model_load_dir))
    if os.path.exists(optim_load_dir) and not config.train_from_scratch:
        optimizer.load_state_dict(torch.load(optim_load_dir))

    best_valid_metric = 0
    test_metric = 0

    for epoch in tqdm_notebook(xrange(config.num_epoch)):
        model.train()
        epoch_loss = 0
        optimizer = lr_decay(optimizer, epoch, config.lr_decay, config.lr)
        random.shuffle(train_batch_list)
        for i in tqdm_notebook(xrange(0, num_train_sample, batch_size)):
            end_idx = min(num_train_sample, i + batch_size)
            cur_batch_list = train_batch_list[i:end_idx]
            cur_batch = batchify(cur_batch_list, gpu=config.gpu)

            optimizer.zero_grad()
            if loss_func is not None:
                output = model(cur_batch)
                loss = loss_func(output, cur_batch['label_tensor'], config.average_batch)
            else:
                loss = model.loss(cur_batch)
            
            loss.backward()
            clip_grad_norm(model.parameters(), config.clip_value)
            optimizer.step()
            epoch_loss += loss

        print "Iteration %2d | Loss: %.6f | var_norm: %.3f | learning rate: %.6f" % (
            epoch + 1,
            epoch_loss / (num_train_sample / batch_size),
            model.parameter_norm(),
            optimizer.param_groups[0]['lr']
        )

#         cur_valid_metric = evaluate(valid_batch_list, model, config)
#         if cur_valid_metric > best_valid_metric:
#             best_valid_metric = cur_valid_metric
#             test_metric = evaluate(test_batch_list, model, config)
#             if not os.path.exists(config.model_dir):
#                 os.makedirs(config.model_dir)
#             torch.save(model.state_dict(), model_save_dir)
#             torch.save(optimizer.state_dict(), optim_save_dir)