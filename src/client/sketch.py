from src.utils.worker_utils import topk
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import numpy as np
from csvec import CSVec
import copy

class SketchClient(object):
    def __init__(self, cid, train_data, test_data, options, model):
        self.D = options['D']
        self.u = torch.zeros(self.D)
        self.v = torch.zeros(self.D)
        self.cid = cid
        self.train_data = train_data
        self.test_data = test_data
        self.num_local_round = options['num_local_round']
        self.num_epoch = options['num_epoch']
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * self.num_local_round * self.num_epoch, shuffle = True)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'] * self.num_local_round * self.num_epoch, shuffle = False)
        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.gpu = options['gpu'] if 'gpu' in options else False

        self.sketch_c = options['c']
        self.sketch_r = options['r']
        self.sketch_k = options['k']
        self.p = options['p']
        self.momentum = options['momentum']

        # make sketch
        self.workersketch = CSVec(d = options['sketchMask'].sum().item(), c = options['c'], r = options['r'])
        self.model = copy.deepcopy(model)
        # self.model = model
        self.move_model_to_gpu(self.model, options)
        self.person_model_params = self.get_flat_model_params()

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

    def sketchHelper(self):
        # sketch v into workersketch
        self.workersketch.accumulateVec(self.v)

    def set_flat_params_to(self, flat_params):
        prev_ind = 0
        for param in self.model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
            )
            prev_ind += flat_size

    def get_flat_model_params(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def get_flat_grads(self):
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.data.view(-1))
        flat_grads = torch.cat(grads)
        return flat_grads

    def zero_out_grad(self):
        for param in self.model.parameters():
            param.grad.data.zero_()

    def local_train(self):
        bytes_w = self.sketch_c * self.sketch_r + self.p * self.sketch_k
        bytes_r = self.sketch_k * 2 + self.p * self.sketch_k

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        flat_grad = torch.zeros(self.D)
        for x, y in self.train_dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            if torch.isnan(pred.max()):
                from IPython import embed
                embed()
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            flat_grad += self.get_flat_grads()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            train_netloss += loss.item() * y.size(0)
            train_loss = train_netloss
            train_acc += correct
            train_total += target_size

            self.zero_out_grad()

        
        self.u = self.momentum * self.u + flat_grad
        self.v += self.u
        self.sketchHelper()

        self.person_model_params = self.get_flat_model_params()
        param_dict = {'norm': torch.norm(self.person_model_params).item(),
            'max': self.person_model_params.max().item(),
            'min': self.person_model_params.min().item()}
        return_dict = {'loss': train_loss / train_total,
            'netloss': train_netloss / train_total,
            'acc': train_acc / train_total}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
            'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return self.workersketch, stats

    def local_test(self, use_eval_data = True):
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        self.model.eval()
        test_loss = test_netloss = test_acc = test_total = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
            
                pred = self.model(x)
                loss = self.criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                test_netloss += loss.item() * y.size(0)
                test_acc += correct
                test_total += target_size
                test_loss = test_netloss
        
        test_dict = {'loss': test_loss, 'netloss': test_netloss,
            'acc': test_acc, 'test_num': len(dataset)}

        return test_dict