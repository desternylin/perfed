import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from src.model.model import choose_model
from src.client.clientbase import Client
import math

class QSGDClient(Client):
    def __init__(self, cid, train_data, valid_data, test_data, options):
        self.cid = cid
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = True)
        self.valid_dataloader = DataLoader(valid_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = False)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = False)
        self.iter_trainloader = iter(self.train_dataloader)
        self.iter_testloader = iter(self.test_dataloader)

        self.model = choose_model(options)
        self.move_model_to_gpu(self.model, options)

        self.local_model = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()

        self.num_q_level = options['num_q_level']
        self.bucket_size = options['bucket_size']

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.gpu = options['gpu'] if 'gpu' in options else False

    def set_local_model_params(self, local_model):
        self.local_model = local_model
        self.set_flat_params_to(self.local_model)

    def inverse_prop_decay_learning_rate(self, round_i):
        self.local_lr /= round_i + 1

    def quantize(self, grad):
        num_bucket = math.ceil(self.local_model_bytes / self.bucket_size)
        if num_bucket == 1:
            self.bucket_size = self.local_model_byte

        chunked_grad = [grad[i:i + self.bucket_size] for i in range(0, self.local_model_bytes, self.bucket_size)]
        tilde_chunked_grad = list()

        for buck_grad in chunked_grad:
            x = buck_grad.float()
            x_norm = torch.norm(x, p = float('inf'))

            sgn_x = ((x > 0).float() - 0.5) * 2

            p = torch.div(torch.abs(x), x_norm)
            renormalize_p = torch.mul(p, self.num_q_level - 1)
            floor_p = torch.floor(renormalize_p)
            compare = torch.rand_like(floor_p)
            final_p = renormalize_p - floor_p
            margin = (compare < final_p).float()
            xi = (floor_p + margin) / (self.num_q_level - 1)

            tilde_chunked_grad.append(x_norm * sgn_x * xi)

        tilde_grad = torch.cat(tilde_chunked_grad)

        return tilde_grad

    def local_train(self):
        bytes_w = (self.local_model_bytes + math.ceil(self.local_model_bytes / self.bucket_size) * 32 + math.ceil(math.log(self.num_q_level, 2)) * self.local_model_bytes) / 32
        bytes_r = bytes_w

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        
        x, y = self.get_next_train_batch()
        if self.gpu:
            x, y = x.cuda(), y.cuda()
        pred = self.model(x)
        if torch.isnan(pred.max()):
            from IPython import embed
            embed()
        loss = self.criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

        grads = list()
        params = []
        for param in self.model.parameters():
            grads.append(param.grad.data.view(-1))
        flat_grads = torch.cat(grads)
        tilde_grads = self.quantize(flat_grads)

        _, predicted = torch.max(pred, 1)
        train_acc = predicted.eq(y).sum().item()
        train_total = y.size(0)
        train_netloss = loss.item() * y.size(0)
        train_loss = train_netloss

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
        return (len(self.train_data), tilde_grads), stats

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


    