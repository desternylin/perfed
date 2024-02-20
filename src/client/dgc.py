import numpy as np
import torch
import time
from src.model.model import choose_model
from torch.utils.data import DataLoader
import torch.nn as nn
from src.client.clientbase import Client
from src.optimizer.fedoptimizer import grad_desc

class DGCClient(Client):
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
        self.optimizer = grad_desc(self.model.parameters(), lr = options['person_lr'])
        self.move_model_to_gpu(self.model, options)

        self.local_model = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()

        self.momentum = options['momentum']
        self.sparse_level = options['sparse_level']

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.gpu = options['gpu'] if 'gpu' in options else False
        total_device = torch.cuda.device_count()
        if self.gpu:
            self.device = 0 if 'device' not in options else ((options['device'] + self.cid % total_device) % total_device)
        else:
            self.device = 'cpu'

        self.uk = torch.zeros_like(self.local_model)
        self.vk = torch.zeros_like(self.local_model)

    def set_local_model_params(self, local_model):
        self.local_model = local_model
        self.set_flat_params_to(self.local_model)

    def sparsify(self, flat_grads):
        self.uk = self.momentum * self.uk + flat_grads
        self.vk += self.uk
        tilde_grads = torch.zeros_like(self.uk)

        prev_ind = 0
        for param in self.model.parameters():
            flat_size = int(np.prod(list(param.size())))
            tmp_vk = self.vk[prev_ind:prev_ind + flat_size]
            thr_ind = round((1 - self.sparse_level) * flat_size)
            if thr_ind > 0:
                thr, _ = torch.kthvalue(abs(tmp_vk), flat_size - thr_ind + 1)
                mask = (abs(tmp_vk) >= thr)
            else:
                mask = torch.zeros(flat_size, dtype = torch.bool)
                if self.gpu:
                    # mask = mask.cuda()
                    mask = mask.to(self.device)
            tilde_grads[prev_ind:prev_ind + flat_size] = tmp_vk * mask
            self.vk[prev_ind:prev_ind + flat_size] *= ~mask
            self.uk[prev_ind:prev_ind + flat_size] *= ~mask

            prev_ind += flat_size
        
        return tilde_grads


    def local_train(self):
        # The compressed gradient requires 32-bit of nonzero gradient values and 16-bit run length of zeros
        bytes_w = (round((1 - self.sparse_level) * self.local_model_bytes) * (32 + 16)) / 32
        bytes_r = bytes_w

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0

        x, y = self.get_next_train_batch()
        if self.gpu:
            x, y = x.to(self.device), y.to(self.device)
            # x, y = x.cuda(), y.cuda()
        pred = self.model(x)
        if torch.isnan(pred.max()):
            from IPython import embed
            embed()
        loss = self.criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

        grads = list()
        for param in self.model.parameters():
            grads.append(param.grad.data.view(-1))
        flat_grads = torch.cat(grads)
        tilde_grads = self.sparsify(flat_grads)

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
        return [len(self.train_data), tilde_grads], stats
    
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
        self.optimizer.step(beta = 0.001)
    
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