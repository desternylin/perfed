import pickle
import json
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['mkdir', 'read_data', 'Metrics', 'MiniDataset', 'topk']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_data(train_data_dir, valid_data_dir, test_data_dir, key=None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    train_data = {}
    valid_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if key is not None:
        train_files = list(filter(lambda x: str(key) in x, train_files))

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_x = np.array(v['x'])
        print('data.shape = {}'.format(train_x.shape))
        train_data[cid] = MiniDataset(v['x'], v['y'])

    valid_files = os.listdir(valid_data_dir)
    valid_files = [f for f in valid_files if f.endswith('.pkl')]
    if key is not None:
        valid_files = list(filter(lambda x: str(key) in x, valid_files))
    
    for f in valid_files:
        file_path = os.path.join(valid_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        valid_data.update(cdata['user_data'])
        
    for cid, v in valid_data.items():
        valid_data[cid] = MiniDataset(v['x'], v['y'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if key is not None:
        test_files = list(filter(lambda x: str(key) in x, test_files))

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, train_data, valid_data, test_data

class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3 and self.data.shape[1] == 28:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1 and self.data.shape[1] == 28:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3 and self.data.shape[1] == 28:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[1] == 32:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        elif self.data.ndim == 4 and self.data.shape[2] == 178:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.CenterCrop((178, 178)),
                transforms.Resize((128, 128)),
                transforms.ToTensor()]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target

class Metrics(object):
    def __init__(self, clients, options, name=''):
        self.options = options
        num_rounds = options['num_round'] + 1
        self.bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.bytes_read = {c.cid: [0] * num_rounds for c in clients}
        self.local_bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.local_bytes_read = {c.cid: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.netloss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.global_loss_on_train_data = [0] * num_rounds
        self.global_netloss_on_train_data = [0] * num_rounds
        self.global_acc_on_train_data = [0] * num_rounds
        self.loss_var_on_train_data = [0] * num_rounds
        self.netloss_var_on_train_data = [0] * num_rounds
        self.acc_var_on_train_data = [0] * num_rounds
        self.num_samples_on_train_data = [0] * num_rounds

        # Statistics in valid procedure
        self.loss_on_valid_data = [0] * num_rounds
        self.netloss_on_valid_data = [0] * num_rounds
        self.acc_on_valid_data = [0] * num_rounds
        self.global_loss_on_valid_data = [0] * num_rounds
        self.global_netloss_on_valid_data = [0] * num_rounds
        self.global_acc_on_valid_data = [0] * num_rounds
        self.loss_var_on_valid_data = [0] * num_rounds
        self.netloss_var_on_valid_data = [0] * num_rounds
        self.acc_var_on_valid_data = [0] * num_rounds
        self.num_samples_on_valid_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.netloss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds
        self.global_loss_on_eval_data = [0] * num_rounds
        self.global_netloss_on_eval_data = [0] * num_rounds
        self.global_acc_on_eval_data = [0] * num_rounds
        self.loss_var_on_eval_data = [0] * num_rounds
        self.netloss_var_on_eval_data = [0] * num_rounds
        self.acc_var_on_eval_data = [0] * num_rounds
        self.num_samples_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        suffix = '{}_sd{}_lr{}_plr{}_lam{}_intd{}_c{}_r{}_k{}_lay{}_s{}_mfr{}_atk{}_agr{}_locr{}_beta{}_delta{}_q{}_buck{}_sp{}_rl{}'.format(name,
                                                    options['seed'],
                                                    options['local_lr'],
                                                    options['person_lr'],
                                                    options['lamda'],
                                                    options['d'],
                                                    options['c'],
                                                    options['r'],
                                                    options['k'],
                                                    options['num_layers_keep'],
                                                    options['server'],
                                                    options['mali_frac'],
                                                    options['attack'],
                                                    options['aggr'],
                                                    options['num_local_round'],
                                                    options['beta'],
													options['delta_thre'],
                                                    options['num_q_level'],
                                                    options['bucket_size'],
                                                    options['sparse_level'],
                                                    options['rising_level'])

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        if options['dis']:
            suffix = options['dis']
            self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        valid_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'valid.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.valid_writer = SummaryWriter(valid_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, bytes_r, = \
            stats['id'], stats['bytes_w'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.netloss_on_train_data[round_i] = train_stats['netloss']
        self.acc_on_train_data[round_i] = train_stats['acc']

        self.num_samples_on_train_data[round_i] = train_stats['num_samples']
        num_samples = train_stats['num_samples']
        self.global_loss_on_train_data[round_i] = sum(train_stats['loss']) / sum(num_samples)
        self.global_netloss_on_train_data[round_i] = sum(train_stats['netloss']) / sum(num_samples)
        self.global_acc_on_train_data[round_i] = sum(train_stats['acc']) / sum(num_samples)

        self.loss_var_on_train_data[round_i] = np.var(np.array(train_stats['loss']) / np.array(num_samples))
        self.netloss_var_on_train_data[round_i] = np.var(np.array(train_stats['netloss']) / np.array(num_samples))
        self.acc_var_on_train_data[round_i] = np.var(np.array(train_stats['acc']) / np.array(num_samples))

        self.train_writer.add_scalar('train_loss', self.global_loss_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_netloss', self.global_netloss_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_acc', self.global_acc_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_loss_var', self.loss_var_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_netloss_var', self.netloss_var_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_acc_var', self.acc_var_on_train_data[round_i], round_i)

    def update_valid_stats(self, round_i, valid_stats):
        self.loss_on_valid_data[round_i] = valid_stats['loss']
        self.netloss_on_valid_data[round_i] = valid_stats['netloss']
        self.acc_on_valid_data[round_i] = valid_stats['acc']

        self.num_samples_on_valid_data[round_i] = valid_stats['num_samples']
        num_samples = valid_stats['num_samples']
        self.global_loss_on_valid_data[round_i] = sum(valid_stats['loss']) / sum(num_samples)
        self.global_netloss_on_valid_data[round_i] = sum(valid_stats['netloss']) / sum(num_samples)
        self.global_acc_on_valid_data[round_i] = sum(valid_stats['acc']) / sum(num_samples)

        self.loss_var_on_valid_data[round_i] = np.var(np.array(valid_stats['loss']) / np.array(num_samples))
        self.netloss_var_on_valid_data[round_i] = np.var(np.array(valid_stats['netloss']) / np.array(num_samples))
        self.acc_var_on_valid_data[round_i] = np.var(np.array(valid_stats['acc']) / np.array(num_samples))

        self.valid_writer.add_scalar('valid_loss', self.global_loss_on_valid_data[round_i], round_i)
        self.valid_writer.add_scalar('valid_netloss', self.global_netloss_on_valid_data[round_i], round_i)
        self.valid_writer.add_scalar('valid_acc', self.global_acc_on_valid_data[round_i], round_i)
        self.valid_writer.add_scalar('valid_loss_var', self.loss_var_on_valid_data[round_i], round_i)
        self.valid_writer.add_scalar('valid_netloss_var', self.netloss_var_on_valid_data[round_i], round_i)
        self.valid_writer.add_scalar('valid_acc_var', self.acc_var_on_valid_data[round_i], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.netloss_on_eval_data[round_i] = eval_stats['netloss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.num_samples_on_eval_data[round_i] = eval_stats['num_samples']
        num_samples = eval_stats['num_samples']
        self.global_loss_on_eval_data[round_i] = sum(eval_stats['loss']) / sum(num_samples)
        self.global_netloss_on_eval_data[round_i] = sum(eval_stats['netloss']) / sum(num_samples)
        self.global_acc_on_eval_data[round_i] = sum(eval_stats['acc']) / sum(num_samples)

        self.loss_var_on_eval_data[round_i] = np.var(np.array(eval_stats['loss']) / np.array(num_samples))
        self.netloss_var_on_eval_data[round_i] = np.var(np.array(eval_stats['netloss']) / np.array(num_samples))
        self.acc_var_on_eval_data[round_i] = np.var(np.array(eval_stats['acc']) / np.array(num_samples))

        self.eval_writer.add_scalar('test_loss', self.global_loss_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('test_netloss', self.global_netloss_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('test_acc', self.global_acc_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('test_loss_var', self.loss_var_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('test_netloss_var', self.netloss_var_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('test_acc_var', self.acc_var_on_eval_data[round_i], round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_every'] = self.options['eval_every']
        metrics['local_lr'] = self.options['local_lr']
        metrics['num_round'] = self.options['num_round']
        metrics['batch_size'] = self.options['batch_size']
        metrics['lamda'] = self.options['lamda']
        metrics['q'] = self.options['q']
        metrics['d'] = self.options['d']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['netloss_on_train_data'] = self.netloss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['global_loss_on_train_data'] = self.global_loss_on_train_data
        metrics['global_netloss_on_train_data'] = self.global_netloss_on_train_data
        metrics['global_acc_on_train_data'] = self.global_acc_on_train_data
        metrics['loss_var_on_train_data'] = self.loss_var_on_train_data
        metrics['netloss_var_on_train_data'] = self.netloss_var_on_train_data
        metrics['acc_var_on_train_data'] = self.acc_var_on_train_data
        metrics['num_samples_on_train_data'] = self.num_samples_on_train_data

        metrics['loss_on_valid_data'] = self.loss_on_valid_data
        metrics['netloss_on_valid_data'] = self.netloss_on_valid_data
        metrics['acc_on_valid_data'] = self.acc_on_valid_data
        metrics['global_loss_on_valid_data'] = self.global_loss_on_valid_data
        metrics['global_netloss_on_valid_data'] = self.global_netloss_on_valid_data
        metrics['global_acc_on_valid_data'] = self.global_acc_on_valid_data
        metrics['loss_var_on_valid_data'] = self.loss_var_on_valid_data
        metrics['netloss_var_on_valid_data'] = self.netloss_var_on_valid_data
        metrics['acc_var_on_valid_data'] = self.acc_var_on_valid_data
        metrics['num_samples_on_valid_data'] = self.num_samples_on_valid_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['netloss_on_eval_data'] = self.netloss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data
        metrics['global_loss_on_eval_data'] = self.global_loss_on_eval_data
        metrics['global_netloss_on_eval_data'] = self.global_netloss_on_eval_data
        metrics['global_acc_on_eval_data'] = self.global_acc_on_eval_data
        metrics['loss_var_on_eval_data'] = self.loss_var_on_eval_data
        metrics['netloss_var_on_eval_data'] = self.netloss_var_on_eval_data
        metrics['acc_var_on_eval_data'] = self.acc_var_on_eval_data
        metrics['num_samples_on_eval_data'] = self.num_samples_on_eval_data

        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as output_file:
            json.dump(str(metrics), output_file)

def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)

    topkIndices = torch.sort(vec**2)[1][-k:]

    ret[topkIndices] = vec[topkIndices]
    return ret

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[0, :] = vv[0, :].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[j, :].clone()
            uk = uk + projection(uj, vk)
        uu[k, :] = vk - uk
    for k in range(nk):
        uk = uu[k, :].clone()
        uu[k, :] = uk / uk.norm()
    return uu
