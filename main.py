import numpy as np
import argparse
import importlib
import torch
import os

from src.utils.worker_utils import read_data, gram_schmidt
from config import OPTIMIZERS, DATASETS, MODELS, MODEL_PARAMS, ALGORITHMS, CRITERIA, ATTACKS, SERVERS, SERVERTYPE, AGGR
from src.server.server import Server
from src.model.model import choose_model
import torch.nn.functional as F

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', 
                        help = 'name of algorithm;',
                        type = str,
                        choices = OPTIMIZERS,
                        default = 'me')
    parser.add_argument('--dataset',
                        help = 'name of dataset;',
                        type = str,
                        default = 'synthetic_alpha0_beta0_niid_balance')
    parser.add_argument('--model', 
                        help = 'name of model;',
                        type = str, 
                        choices = MODELS,
                        default = 'logistic')
    parser.add_argument('--criterion',
                        help = 'name of loss function',
                        type = str,
                        choices = CRITERIA,
                        default = 'celoss')
    parser.add_argument('--wd',
                        help = 'weight decay parameter;',
                        type = float,
                        default = 0.001)
    parser.add_argument('--gpu',
                        help = 'use gpu (default: False)',
                        default = False,
                        action = 'store_true')
    parser.add_argument('--noprint',
                        help = 'whether to print inner result (default: False)',
                        default = False,
                        action = 'store_true')
    parser.add_argument('--simpleaverage',
                        help = 'whether to use simple average or weighted average for local solutions (default: True, i.e. simple average)',
                        default = True,
                        action = 'store_false')
    parser.add_argument('--device', 
                        help = 'selected CUDA device',
                        default = 0,
                        type = int)
    parser.add_argument('--num_round',
                        help = 'number of rounds to simulate',
                        type = int, 
                        default = 200)
    parser.add_argument('--eval_every',
                        help = 'evaluate every ___ rounds',
                        type = int,
                        default = 5)
    parser.add_argument('--clients_per_round',
                        help = 'number of clients selected per round',
                        type = int,
                        default = 10)
    parser.add_argument('--batch_size',
                        help = 'batch size when clients train on data',
                        type = int,
                        default = 64)
    parser.add_argument('--num_epoch',
                        help = 'number of rounds for solving the personalization sub-problem when clients train on data',
                        type = int,
                        default = 20)
    parser.add_argument('--num_local_round',
                        help = 'number of local rounds for local update',
                        type = int,
                        default = 5)
    parser.add_argument('--local_lr',
                        help = 'learning rate for local update',
                        type = float,
                        default = 0.1)
    parser.add_argument('--person_lr',
                        help = 'learning rate for personalization sub-problem',
                        type = float,
                        default = 0.1)
    parser.add_argument('--lamda',
                        help = 'regularization tuning parameter for personalization',
                        type = float,
                        default = 15)
    parser.add_argument('--q',
                        help = 'parameter for controlling fairness',
                        type = float,
                        default = 1.0)
    parser.add_argument('--d',
                        help = 'dimension for the global model',
                        type = int,
                        default = 10)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parser.add_argument('--server',
                        help = 'type of server',
                        type = str,
                        default = 'server',
                        choices = SERVERS)
    parser.add_argument('--mali_frac',
                        help = 'fraction of malicious clients for robustness testing',
                        type = float,
                        default = 0.2)
    parser.add_argument('--attack',
                        help = 'type of the attack',
                        type = str,
                        default = 'same_value',
                        choices = ATTACKS)
    parser.add_argument('--p',
                        help = 'lp norm',
                        type = float,
                        default = 1.0)
    parser.add_argument('--aggr',
                        help = 'aggregate type',
                        type = str,
                        default = 'mean',
                        choices = AGGR)
    parser.add_argument('--sketchparamslargerthan',
                        help = 'sketch everything larger than sketchparamslargerthan',
                        type = int,
                        default = 0)
    parser.add_argument('--p2',
                        help = 'request p2 times k exact values from the clients',
                        default = 4,
                        type = int)
    parser.add_argument('--c',
                        help = 'columns in the sketch',
                        type = float,
                        default = 40)
    parser.add_argument('--r',
                        help = 'rows in the sketch',
                        type = float,
                        default = 7)
    parser.add_argument('--k',
                        help = 'generate k-sparse vector from the sketch',
                        type = float,
                        default = 10)
    parser.add_argument('--momentum',
                        help = 'momentum when doing update in sketched-sgd and dgc',
                        type = float,
                        default = 0.9)
    parser.add_argument('--sketchbiases',
                        help = 'whether sketch bias term in the model',
                        default = True,
                        action = 'store_false')
    parser.add_argument('--doaccumulateerror',
                        help = 'whether accumulate error',
                        default = True,
                        action = 'store_false')
    parser.add_argument('--num_layers_keep',
                        help = 'number of layers to use for the global model',
                        type = int,
                        default = 0)
    parser.add_argument('--beta',
                        help = 'parameter to control the aggregation',
                        type = float,
                        default = 1)
    parser.add_argument('--orthogonalize',
                        help = 'whether to orthogonalize the projection matrix',
                        default = False,
                        action = 'store_true')
    parser.add_argument('--delta_thre',
                        help = 'look-back phase error threshold',
                        type = float,
                        default = 0.2)
    parser.add_argument('--num_q_level',
                        help = 'number of quantization levels',
                        type = int,
                        default = 5)
    parser.add_argument('--bucket_size',
                        help = 'bucket size for gradient quantization',
                        type = int,
                        default = 5)
    parser.add_argument('--sparse_level',
                        help = 'initial sparse level for dgc',
                        type = float,
                        default = 0.75)
    parser.add_argument('--rising_level',
                        help = 'rising level for dgc during warm-up training',
                        type = float, 
                        default = 0.25)

    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)


    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    # If using projection, setup the projection matrix
    proj_algo = ['proj', 'proj_fair', 'lp_proj', 'lp_projnew', 'lp_projdiff']
    tmp_model = choose_model(options)
    person_model_dim = sum(p.numel() for p in tmp_model.parameters())
    print('Total number of parameters is {}'.format(person_model_dim))
    if options['algo'] in proj_algo:
        Proj = torch.normal(mean = 0., std = 1., size = (options['d'], person_model_dim))
        Proj = F.normalize(Proj, p = 2, dim = 1)
        if options['orthogonalize']:
            Proj = gram_schmidt(Proj)
        options.update({'Proj': Proj, 'local_model_dim': options['d'], 'person_model_dim': person_model_dim})
    else:
        options.update({'local_model_dim': person_model_dim, 'person_model_dim': person_model_dim})
    del(tmp_model)

    # Load selected server
    server_path = 'src.server.%s' % options['server']
    mod = importlib.import_module(server_path)
    server_class = getattr(mod, SERVERTYPE[options['server']])

    return options, server_class, dataset_name, sub_data

def main():
    # Parse command line arguments
    options, server_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    valid_path = os.path.join('./data', dataset_name, 'data', 'valid')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, train_data, valid_data, test_data)
    all_data_info = read_data(train_path, valid_path, test_path, sub_data)

    # Call appropriate server
    selected_server = server_class(all_data_info, options)
    selected_server.train()

if __name__ == '__main__':
    main()