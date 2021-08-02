from src.optimizer.fedoptimizer import grad_desc
from src.model.model import choose_model
import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class FedAvgClient(object):
    def __init__(self, cid, train_data, test_data, options):
        self.cid = cid
        self.train_data = train_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = True)
        self.test_dataloader = DataLoader(test_data, batch_size = options['batch_size'] * options['num_epoch'], shuffle = False)

        self.model = choose_model(options)
        self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])
        self.move_model_to_gpu(self.model, options)

        self.local_model = self.get_flat_model_params()
        self.local_model_bytes = self.local_model.numel()

        self.local_lr = options['local_lr']
        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
        self.num_local_round = options['num_local_round']
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
        self.set_flat_params_to(self.local_model)

    def inverse_prop_decay_learning_rate(self, round_i):
        self.local_lr /= round_i + 1

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

    def local_train(self):
        bytes_w = self.local_model_bytes
        bytes_r = self.local_model_bytes

        begin_time = time.time()

        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        # print('>>> Client {} local_model.norm = {}'.format(self.cid, torch.norm(self.local_model)))
        for local_round in range(self.num_local_round):
            last_train_loss = train_loss
            train_loss = train_netloss = train_acc = train_total = 0.0
            for x, y in self.train_dataloader:
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
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_netloss += loss.item() * y.size(0)
                train_loss = train_netloss
                train_acc += correct
                train_total += target_size

            # if train_loss - last_train_loss < 1e-20 and local_round > 1:
            #     break
        
        self.local_model = self.get_flat_model_params()
        # print('>>> Client {} updated local_model.norm = {}'.format(self.cid, torch.norm(self.local_model)))
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
        # test_dict = {'loss': test_loss, 'netloss': test_netloss,
        #     'acc': test_acc, 'test_num': len(dataset)}
        test_dict = {'loss': test_loss, 'netloss': test_netloss,
            'acc': test_acc, 'test_num': test_total}

        return test_dict

    def get_flat_grads(self, dataloader):
        flat_grads = torch.zeros_like(self.local_model)
        return flat_grads

    def get_loss(self, dataloader):
        self.model.eval()
        loss, total_num = 0.0, 0
        for x, y in dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += self.criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        return loss

    def solve_grad(self):
        bytes_w = self.local_model_bytes
        bytes_r = self.local_model_bytes
        train_loss = self.get_loss(self.train_dataloader)
        test_loss = self.get_loss(self.test_dataloader)

        stats = {'id': self.cid, 'bytes_w': bytes_w,
        'bytes_r': bytes_r,
        'train_loss': train_loss, 'test_loss': test_loss}

        grads = self.get_flat_grads(self.train_dataloader)
        grads_np = grads.cpu().detach().numpy()

        return (len(self.train_data), grads_np), stats