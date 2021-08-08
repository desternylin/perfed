from src.client.clientbase import Client
from src.optimizer.fedoptimizer import lp_proj_optimizer
from src.model.model import choose_model
import torch
import time

class LpProjClient(Client):
    def __init__(self, cid, train_data, test_data, options):
        self.model = choose_model(options)
        self.optimizer = lp_proj_optimizer(self.model.parameters(), p = options['p'], Proj = options['Proj'], lr = options['person_lr'], lamda = options['lamda'], weight_decay = options['wd'])
        self.lamda = options['lamda']
        self.p = options['p']
        self.Proj = options['Proj']
        self.move_model_to_gpu(self.model, options)
        local_model_dim = options['d']
        self.local_model = torch.zeros(local_model_dim)

        super(LpProjClient, self).__init__(cid, train_data, test_data, options)

    def local_train(self):
        bytes_w = self.local_model_bytes
        begin_time = time.time()

        for local_round in range(self.num_local_round):
            person_stats = self.person_train()
            proj_person_weight = torch.matmul(self.Proj, self.person_model_params)
            conv_weight = self.local_model - proj_person_weight
            self.local_model = self.local_model - self.local_lr * self.lamda * torch.sign(conv_weight) * (torch.abs(conv_weight) / torch.norm(conv_weight, p = self.p))**(self.p - 1)

        end_time = time.time()
        bytes_r = self.local_model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
            'time': round(end_time - begin_time, 2)}
        stats.update(person_stats)

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

        regularizer = self.lamda * torch.norm(self.local_model - torch.matmul(self.Proj.data, self.person_model_params), p = self.p)
        test_loss = (test_netloss + regularizer * test_total).item()
        test_dict = {'loss': test_loss, 'netloss': test_netloss,
            'acc': test_acc, 'test_num': len(dataset)}

        return test_dict

    # def get_flat_grads(self, dataloader):
    #     conv_weight = self.local_model - torch.matmul(self.Proj.data, self.person_model_params)
    #     flat_grads = self.lamda * torch.sign(conv_weight) * (torch.abs(conv_weight) / torch.norm(conv_weight, p = self.p))**(self.p - 1)
    #     return flat_grads

    # def get_loss(self, dataloader):
    #     self.model.eval()
    #     loss, total_num = 0.0, 0
    #     for x, y in dataloader:
    #         if self.gpu:
    #             x, y = x.cuda(), y.cuda()
    #         pred = self.model(x)
    #         loss += self.criterion(pred, y) * y.size(0)
    #         total_num += y.size(0)
    #     loss /= total_num

    #     conv_weight = self.local_model - torch.matmul(self.Proj.data, self.person_model_params)
    #     loss += self.lamda * torch.norm(conv_weight, p = self.p)
    #     return loss

    def person_train(self):
        self.model.train()
        train_loss = train_netloss = train_acc = train_total = 0.0
        for epoch in range(self.num_epoch):
            last_train_loss = train_loss
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
            self.optimizer.step(self.local_model)
            self.person_model_params = self.get_flat_model_params()
            conv_weight = self.local_model - torch.matmul(self.Proj.data, self.person_model_params)
            regularizer = self.lamda * torch.norm(conv_weight, p = self.p)

            _, predicted = torch.max(pred, 1)
            train_acc = predicted.eq(y).sum().item()
            train_total = y.size(0)
            train_netloss = loss.item() * y.size(0)
            train_loss = (loss.item() + regularizer.item()) * y.size(0)

            # for x, y in self.train_dataloader:
            #     if self.gpu:
            #         x, y = x.cuda(), y.cuda()
            #     self.optimizer.zero_grad()
            #     pred = self.model(x)
            #     if torch.isnan(pred.max()):
            #         from IPython import embed
            #         embed()
            #     loss = self.criterion(pred, y)
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            #     self.optimizer.step(self.local_model)
            #     self.person_model_params = self.get_flat_model_params()
            #     conv_weight = self.local_model - torch.matmul(self.Proj.data, self.person_model_params)
            #     regularizer = self.lamda * torch.norm(conv_weight, p = self.p)

            #     _, predicted = torch.max(pred, 1)
            #     correct = predicted.eq(y).sum().item()
            #     target_size = y.size(0)
            #     train_netloss += loss.item() * y.size(0)
            #     train_loss += (loss.item() + regularizer.item()) * y.size(0)
            #     train_acc += correct
            #     train_total += target_size
            
            if train_loss - last_train_loss < 1e-20 and epoch > 1:
                break

        param_dict = {'norm': torch.norm(self.person_model_params).item(),
            'max': self.person_model_params.max().item(),
            'min': self.person_model_params.min().item()}
        return_dict = {'loss': train_loss / train_total,
            'netloss':train_netloss / train_total,
            'acc': train_acc / train_total}
        return_dict.update(param_dict)
        return return_dict