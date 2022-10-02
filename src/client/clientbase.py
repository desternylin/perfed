import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

class Client(object):
    def __init__(self, cid, train_data, valid_data, test_data, options):
        self.cid = cid
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'], shuffle = True)
        self.valid_dataloader = DataLoader(valid_data, batch_size = options['batch_size'], shuffle = False)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'], shuffle = False)
        self.iter_trainloader = iter(self.train_dataloader)
        self.iter_testloader = iter(self.test_dataloader)

        self.person_model_params = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()
        
        self.local_lr = options['local_lr']
        self.lamda = options['lamda']
        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.num_local_round = options['num_local_round']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Do not use gpu')

    def set_local_model_params(self, local_model):
        self.local_model = local_model

    def inverse_prop_decay_learning_rate(self, round_i):
        self.local_lr /= round_i + 1

    def print_layer_norm(self, local_model):
        total_layer_num = len(self.model.weight_keys)
        layer_param_num = []
        model_dict = self.model.state_dict()
        for layer_keys in self.model.weight_keys:
            tmp_num = 0
            for key in model_dict.keys():
                if key in layer_keys:
                    tmp_num += model_dict[key].numel()
            layer_param_num.append(tmp_num)
        print('>>> layer_param_num = {}'.format(layer_param_num))
        layer_model = torch.split(local_model, layer_param_num)
        layer_norm = []
        for layer in range(len(layer_model)):
            layer_norm.append(round(torch.norm(layer_model[layer]).item(), 3))
        print('>>> layer_norm = {}'.format(layer_norm))

    def get_flat_model_params(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def set_flat_params_to(self, flat_params):
        prev_ind = 0
        for param in self.model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
            )
            prev_ind += flat_size

    def get_next_train_batch(self):
        try:
            (x, y) = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.train_dataloader)
            (x, y) = next(self.iter_trainloader)
        return (x, y)

    def get_next_test_batch(self):
        try:
            (x, y) = next(self.iter_testloader)
        except StopIteration:
            self.iter_testloader = iter(self.test_dataloader)
            (x, y) = next(self.iter_testloader)
        return (x, y)