import numpy as np
import time
import os
import argparse
from train import train
from zoopt import Dimension, Objective, Parameter, Opt, Solution
from zoopt.utils.zoo_global import gl

def setting(edge_dir):
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The directory of data")
    parser.add_argument("--edge_dir", type=str, default="./edge/time_space_lateral",
                        help="The directory of edge and binary feature")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="The number of labels(2 or 5)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=2,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--smooth_eps", type=float, default=0.0,
                        help="label smooth epsilon")
    parser.add_argument("--optim_type", type=str, default='adam',
                        help="Optimizer(adam/sgd)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum in sgd")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="parameter in optim scheduler")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    
    return args

def solution_to_hyper_param(x, args):
    if not isinstance(x, list):
        x = x.get_x()
    args.num_heads = 2**x[0]
    args.num_out_heads = 2**x[1]
    args.num_layers = x[2]
    args.num_hidden = 2**x[3]
    args.in_drop = x[4]
    args.attn_drop = x[5]
    return args

best_acc = 0
best_hyper_params = None

def valid_loss(x, edge_dir):
    args = setting(edge_dir)
    hyper_param = solution_to_hyper_param(x, args)
    print("cur hyperparameter:", end=" ")
    print(hyper_param)
    value = -train(hyper_param)
    print("cur acc: %.4f" % -value)

    global best_acc
    global best_hyper_params
    if -value > best_acc:
        best_acc = -value
        best_hyper_params = hyper_param

    print("cur best acc: %.4f" % best_acc)
    print("cur best hyperparameter:", end=" ")
    print(best_hyper_params)
    return value

def hyper_param_opt(parser):
    
    gl.set_seed(int(time.time()))
    np.random.seed(int(time.time()))

    num_heads = list(range(0, 5))
    num_out_heads = list(range(0, 4))
    num_layers = list(range(4, 11))
    num_hidden = list(range(3, 6))
    in_drop = [0.0, 0.5]
    attn_drop = [0.0, 0.5]
    
    dim_size = 6
    dim_regs = [num_heads, num_out_heads, num_layers, num_hidden, in_drop, attn_drop]
                
    dim_tys = [False, False, False, False, True, True]

    repeat = 1000
    for j in range(repeat):
        init_sample = []
        for i in range(dim_size):
            if dim_tys[i] == False:
                idx = np.random.randint(low=0, high=len(dim_regs[i]))
                init_sample.append(dim_regs[i][idx])
            else:
                p = np.random.rand() * 0.3
                init_sample.append(p)
        
        valid_loss(init_sample, parser)
#         dim = Dimension(dim_size, dim_regs, dim_tys)
#         objective = Objective(valid_loss, dim)
#         budget = 5
#         parameter = Parameter(budget=budget, init_samples=init_sample)
#         solution = Opt.min(objective, parameter)


if __name__ == "__main__":
    edge_parser = argparse.ArgumentParser(description='GAT')
    edge_parser.add_argument("--edge_dir", type=str, default="./edge/time_space_lateral",
                        help="The directory of edge and binary feature")
    edge_args = edge_parser.parse_args()
    hyper_param_opt(edge_args.edge_dir)
