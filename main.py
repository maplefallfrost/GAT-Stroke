import torch as th
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np

from dataset import StrokeDataset, collate, to_device
from torch.utils.data import DataLoader
from egat import EGAT
from evaluate import evaluate, print_result

def init_model(args):
    if args.gpu >= 0:
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    heads = [args.num_heads] * args.num_layers
    model = EGAT(args.num_layers,
                args.node_f_dim,
                args.edge_f_dim,
                args.num_hidden,
                args.num_classes,
                heads,
                nn.LeakyReLU(args.alpha),
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.temperature,
                args.edge_feature_attn,
                args.edge_update).to(device)
    return model, device

def train(args):
    trainset = StrokeDataset(args.data_dir, args.edge_dir, "train", args.num_classes)
    validset = StrokeDataset(args.data_dir, args.edge_dir, "valid", args.num_classes)
    testset = StrokeDataset(args.data_dir, args.edge_dir, "test", args.num_classes)

    model_dir, _ = os.path.split(args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    model, device = init_model(args)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              drop_last=True)

    valid_loader = DataLoader(validset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate)

    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_valid_acc = 0
    best_test_acc = 0
    best_round = 0

    start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for it, batch in enumerate(train_loader):
            fg, lg = to_device(batch, device)
            logits = model(fg)
            labels = lg.ndata['y']
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (it + 1)
        epoch_duration = time.time() - epoch_start
        print('Epoch: {:3d}, loss: {:4f}, speed: {:.2f}doc/s'.format(
            epoch, epoch_loss, len(trainset) / epoch_duration))

        train_acc, _ = evaluate(model, device, train_loader, args.num_classes, "train")
        valid_acc, _ = evaluate(model, device, valid_loader, args.num_classes, "valid")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_acc, test_conf_mat = evaluate(model, device, test_loader, args.num_classes, "test")
            best_conf_mat = test_conf_mat
            best_round = epoch
            th.save(model.state_dict(), args.model_path)
        
        cur_lr = 0.5 * (1 + np.cos((epoch + 1) * np.pi / args.epochs)) * args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    print("Best round: %d" % best_round)
    print_result(best_conf_mat)
    duration = time.time() - start
    print("Time cost: {:.4f}s".format(duration))

    return test_acc

def multiple_train(args):
    repeat = args.repeat
    results = np.empty(shape=(repeat,))
    for i in range(repeat):
        results[i] = train(args)
    print(results)
    print("mean acc: %.5f%%" % np.mean(results))
    print("std: %.5f%%" % np.std(results))


def test_evaluate(args):
    model, device = init_model(args)
    model.eval()
    model.load_state_dict(th.load(args.model_path))

    testset = StrokeDataset(args.data_dir, args.edge_dir, "test", args.num_classes)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate)
    test_acc, test_conf_mat = evaluate(model, device, test_loader, args.num_classes, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--mode", type=str, required=True,
                        help="mode(train/eval)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The directory of data")
    parser.add_argument("--edge_dir", type=str, default="./edge/time_space",
                        help="The directory of edge and binary feature")
    parser.add_argument("--model_path", type=str, required=True,
                        help="save model path")
    parser.add_argument("--node_f_dim", type=int, default=23,
                        help="The dimension of unary features")
    parser.add_argument("--edge_f_dim", type=int, default=19,
                        help="The dimension of binary features")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="The number of labels(2 or 5)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=75,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature used in attention softmax")
    parser.add_argument("--edge_feature_attn", action="store_true", default=False,
                        help="whether use edge feature in attention")
    parser.add_argument("--edge_update", action="store_true", default=False,
                        help="whether use edge update")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of time for repeating experiments")
    args = parser.parse_args()

    print(args)
    if args.mode == 'train':
        multiple_train(args)
    elif args.mode == 'eval':
        test_evaluate(args)
