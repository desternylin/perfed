import torch
from torch.utils.data import DataLoader
from src.optimizer.fedoptimizer import grad_desc
from src.client.clientbase import Client
import torch.nn as nn
import time
import numpy as np
import copy

class LgClient(Client):
    def __init__(self, cid, train_data, valid_data, test_data, options, model):
        self.cid = cid
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.num_local_round = options['num_local_round']
        self.num_epoch = options['num_epoch']
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * self.num_epoch, shuffle = True)
        self.valid_dataloader = DataLoader(valid_data, batch_size = options['batch_size'] * self.num_epoch, shuffle = False)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'] * self.num_epoch, shuffle = False)
        self.iter_trainloader = iter(self.train_dataloader)
        self.iter_testloader = iter(self.test_dataloader)

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.gpu = options['gpu'] if 'gpu' in options else False
        
        self.model = copy.deepcopy(model)
        self.move_model_to_gpu(self.model, options)
        self.local_model = copy.deepcopy(self.model.state_dict())
        self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])

        self.num_param_glob = options['num_param_glob']
        self.num_param_local = options['num_param_local']

    def local_train(self):
        bytes_w = self.num_param_glob
        bytes_r = self.num_param_glob

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        for local_round in range(self.num_local_round):
            if train_total == 0.0:
                last_train_loss = 0.0
            else:
                last_train_loss = train_loss / train_total
            train_loss = train_netloss = train_acc = train_total = 0.0

            x, y = self.get_next_train_batch()
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            pred = self.model(x)
            if torch.isnan(pred.max()):
                from IPython import embed
                embed()
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            train_acc = predicted.eq(y).sum().item()
            train_total = y.size(0)
            train_netloss = loss.item() * y.size(0)
            train_loss = train_netloss

            if abs(train_loss / train_total - last_train_loss) < 1e-20 and epoch > 1:
                break

        flat_model_params = self.get_flat_model_params()
        param_dict = {'norm': torch.norm(flat_model_params).item(),
            'max': flat_model_params.max().item(),
            'min': flat_model_params.min().item()}
        return_dict = {'loss': train_loss / train_total,
            'netloss': train_netloss / train_total,
            'acc': train_acc / train_total}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
            'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        self.local_model = copy.deepcopy(self.model.state_dict())
        return self.local_model, stats

    def local_test(self, use_eval_data = 2):
        if use_eval_data == 2:
            dataloader, dataset = self.test_dataloader, self.test_data
        elif use_eval_data == 1:
            dataloader, dataset = self.valid_dataloader, self.valid_data
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