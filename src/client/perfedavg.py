from src.client.clientbase import Client
from src.optimizer.fedoptimizer import grad_desc
from src.model.model import choose_model
import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class PerFedAvgClient(Client):
    def __init__(self, cid, train_data, valid_data, test_data, options):
        self.cid = cid
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = True)
        self.valid_dataloader = DataLoader(valid_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = True)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = False)
        self.iter_trainloader = iter(self.train_dataloader)
        self.iter_testloader = iter(self.test_dataloader)

        self.model = choose_model(options)
        self.optimizer = grad_desc(self.model.parameters(), lr = options['person_lr'])
        self.local_lr = options['local_lr']
        self.move_model_to_gpu(self.model, options)

        self.local_model = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.num_local_round = options['num_local_round']
        self.gpu = options['gpu'] if 'gpu' in options else False
        
    def set_local_model_params(self, local_model):
        self.local_model = local_model
        self.set_flat_params_to(self.local_model)

    def local_train(self):
        bytes_w = self.local_model_bytes
        bytes_r = self.local_model_bytes

        begin_time = time.time()

        self.model.train()
        for local_round in range(self.num_local_round):
            tmp_model = self.get_flat_model_params()

            # Step 1
            x, y = self.get_next_train_batch()
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            # Step 2
            x, y = self.get_next_train_batch()
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()

            # restore the model parameters to the one before first update
            self.set_flat_params_to(tmp_model)

            self.optimizer.step(beta = self.local_lr)

            _, predicted = torch.max(pred, 1)
            train_acc = predicted.eq(y).sum().item()
            train_loss = train_netloss = loss.item() * y.size(0)
            train_total = y.size(0)
        
        self.local_model = self.get_flat_model_params()
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        return_dict = {'loss': train_loss / train_total,
            'netloss': train_netloss / train_total,
            'acc': train_acc / train_total}
        return_dict.update(param_dict)
        
        end_time = time.time()
        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
            'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)

        return (len(self.train_data), self.local_model), stats

    def train_one_step(self):
        self.model.train()
        # Step 1
        x, y = self.get_next_train_batch()
        if self.gpu:
            x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        # Step 2
        x, y = self.get_next_train_batch()
        if self.gpu:
            x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step(beta = self.local_lr)
    
    def local_test(self, use_eval_data = 2):
        if use_eval_data == 2:
            dataloader, dataset = self.test_dataloader, self.test_data
        elif use_eval_data == 1:
            dataloader, dataset = self.valid_dataloader, self.valid_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        self.train_one_step()

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
            'acc': test_acc, 'test_num': test_total}

        self.set_flat_params_to(self.local_model)

        return test_dict