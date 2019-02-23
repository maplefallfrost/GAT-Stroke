import torch as th
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse

from dataset import StrokeDataset, collate
from torch.utils.data import DataLoader
from gcn import GCNNet
from gat import GAT
from evaluate import evaluate, print_result


def train(args):
    
    data_dir = args.data_dir
    gpu = args.gpu
    node_f_dim = 23
    edge_f_dim = 19
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_hidden = args.num_hidden
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_layers = args.num_layers
    residual = args.residual
    in_drop = args.in_drop
    attn_drop = args.attn_drop
    lr = args.lr
    weight_decay = args.weight_decay
    alpha = args.alpha
    epochs = args.epochs
    
    if gpu >= 0:
        device = th.device("cuda")
    else:
        device = th.device("cpu")
        
    trainset = StrokeDataset(os.path.join(data_dir, "train"), num_classes)
    validset = StrokeDataset(os.path.join(data_dir, "valid"), num_classes)
    testset = StrokeDataset(os.path.join(data_dir, "test"), num_classes)
        
    train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate(device))

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate(device))

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate(device))
    
    heads = ([num_heads] * num_layers) + [num_out_heads]

    model = GAT(num_layers, node_f_dim, edge_f_dim, num_hidden, num_classes,
                heads, F.elu, in_drop, attn_drop, alpha, residual).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    epoch_losses = []
    best_valid_acc = 0
    best_test_acc = 0
    best_round = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for it, (fg, lg) in enumerate(train_loader):
            logits = model(fg)
            labels = lg.ndata['y']
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (it + 1)
        print('Epoch {:3d}, loss {:4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        train_acc, _= evaluate(model, train_loader, num_classes, "train")
        valid_acc, _ = evaluate(model, valid_loader, num_classes, "valid")
        if valid_acc > best_valid_acc:
            test_acc, test_conf_mat = evaluate(model, test_loader, num_classes, "test")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_conf_mat = test_conf_mat
                best_round = epoch
                
        scheduler.step(valid_acc)
        cur_learning_rate = optimizer.param_groups[0]['lr']
        print('Learning rate: {:10f}'.format(cur_learning_rate))
        if cur_learning_rate <= 1e-6:
            break

    print("Best round: %d" % best_round)
    print_result(best_conf_mat)
    
    return best_test_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The directory of data")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="The number of labels(2 or 5)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=8,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.0,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.2,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    
    print(args)
    train(args)