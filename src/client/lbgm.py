from src.optimizer.fedoptimizer import grad_desc
from src.model.model import choose_model
from src.client.clientbase import Client
import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class LBGMClient(Client):
    def __init__(self, cid, train_data, valid_data, test_data, options):
        self.cid = cid
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = True)
        self.valid_dataloader = DataLoader(valid_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = False)
        self.test_dataloader = DataLoader(test_data, batch_size = options['num_epoch'] * options['num_epoch'], shuffle = False)
        self.iter_trainloader = iter(self.train_dataloader)
        self.iter_testloader = iter(self.test_dataloader)
        self.gpu = options['gpu'] if 'gpu' in options else False
        self.device = 0 if 'device' not in options else options['device']

        self.model = choose_model(options)
        self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])
        self.move_model_to_gpu(self.model, options)
        self.LBG = self.get_flat_model_grads()
        if self.gpu:
            self.LBG = self.LBG.cuda()

        self.delta_thre = options['delta_thre']

        self.local_model = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()

        self.local_lr = options['local_lr']
        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.num_local_round = options['num_local_round']

    def set_local_model_params(self, local_model):
        self.local_model = local_model
        self.set_flat_params_to(self.local_model)

    def get_flat_model_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(param.grad)

        if None not in grads:
            flat_grads = torch.cat(grads)
        else:
            params = []
            for param in self.model.parameters():
                params.append(param.data.view(-1))
            flat_params = torch.cat(params)
            flat_grads = torch.zeros(len(flat_params))

        return flat_grads

    def compute_LBP_error(self, grad_accum):
        if torch.norm(grad_accum) == 0 or torch.norm(self.LBG) == 0:
            return np.inf
        else:
            coef = torch.square(torch.dot(grad_accum, self.LBG) / (torch.norm(grad_accum) * torch.norm(self.LBG)))
            return 1 - coef.item()

    def local_train(self):
        bytes_r = self.local_model_bytes

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        self.optimizer.zero_grad()
        grad_accum = self.get_flat_model_grads()
        if self.gpu:
            grad_accum = grad_accum.cuda()
        for local_round in range(self.num_local_round):
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

            grad = self.get_flat_model_grads()
            if self.gpu:
                grad = grad.cuda()
            grad_accum += grad

            _, predicted = torch.max(pred, 1)
            train_acc = predicted.eq(y).sum().item()
            train_total = y.size(0)
            train_netloss = loss.item() * y.size(0)
            train_loss = train_netloss

        self.local_model = self.get_flat_model_params()
        param_dict = {'norm':torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        return_dict = {'loss': train_loss / train_total,
            'netloss': train_netloss / train_total,
            'acc': train_acc / train_total}
        return_dict.update(param_dict)

        end_time = time.time()
        LBP_err = self.compute_LBP_error(grad_accum)
        if LBP_err <= self.delta_thre:
            mu_k = (torch.dot(grad_accum, self.LBG) / torch.norm(self.LBG)).item() * self.LBG
            bytes_w = 1
        else:
            mu_k = grad_accum
            self.LBG = grad_accum
            bytes_w = self.local_model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
            'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), mu_k), stats

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
            'acc': test_acc, 'test_num': test_total}

        return test_dict
            
