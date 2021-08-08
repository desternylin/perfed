import numpy as np
import torch
import time
import importlib
from src.utils.worker_utils import Metrics
from src.client.lg import LgClient
from src.model.model import choose_model
from config import ALGORITHMS
import itertools
import copy

class ServerLg(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.all_train_data_num = 0
        self.model = choose_model(options)
        self.local_lr = options['local_lr']
        self.latest_model = copy.deepcopy(self.model.state_dict())
        
        self.num_round = options['num_round']
        self.eval_every = options['eval_every']

        # Issues on representation learning
        self.total_num_layers = len(self.model.weight_keys)
        self.num_layers_keep = options['num_layers_keep']
        assert self.num_layers_keep <= self.total_num_layers
        if self.num_layers_keep == 0:
            w_glob_keys = self.model.weight_keys
        else:
            w_glob_keys = self.model.weight_keys[self.total_num_layers - self.num_layers_keep:]
        self.w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

        self.num_param_glob =  0
        self.num_param_local = 0
        for key in self.latest_model.keys():
            self.num_param_local += self.latest_model[key].numel()
            if key in self.w_glob_keys:
                self.num_param_glob += self.latest_model[key].numel()
        percent_param = 100 * float(self.num_param_glob / self.num_param_local)
        print('# Params: {} (local), {} (global); Percentage {:.2f}'.format(
            self.num_param_local, self.num_param_glob, percent_param
        ))

        # Setup clients
        self.clients = self.setup_clients(dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.clients_per_round = options['clients_per_round']

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']

    def setup_clients(self, dataset, options):
        users, train_data, test_data = dataset
        
        all_clients = []
        for user in users:
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            options['num_param_glob'] = self.num_param_glob
            options['num_param_local'] = self.num_param_local
            c = LgClient(user_id, train_data[user], test_data[user], options, self.model)
            all_clients.append(c)

        return all_clients

    def update_local_model(self, local_model):
        for key in self.w_glob_keys:
            local_model[key] = self.latest_model[key]
        return local_model

    def select_clients(self, seed = 1):
        """Selects num_clients clients from possible clients"""
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace = False).tolist()

    def get_flat_model_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        for round_i in range(self.num_round):
            print('>>> Round {}'.format(round_i))

            # Broadcast the latest model to all the clients
            for c in self.clients:
                local_model = self.update_local_model(c.local_model)
                c.model.load_state_dict(local_model)
                # # Just for debugging
                # flat_local_model = self.get_flat_model_params(c.model)
                # print('norm of client {} local_model = {}'.format(c.cid, torch.norm(flat_local_model)))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # choose K clients for the aggregation
            selected_clients = self.select_clients(seed = round_i)

            # Do local update for the selected clients
            stats = []
            for i, c in enumerate(selected_clients, start = 1):
                soln, stat = c.local_train()
                # # Just for debugging
                # flat_soln = self.get_flat_model_params(c.model)
                # print('>>> norm of client {} soln = {}'.format(c.cid, torch.norm(flat_soln)))

                if i == 1:
                    self.latest_model = copy.deepcopy(soln)
                else:
                    for key in self.w_glob_keys:
                        self.latest_model[key] += soln[key]
                
                if self.print_result:
                    print('Round: {: >2d} | CID:{: >3d}({:>2d}/{:>2d})|'
                    'loss {:>.4f} | Acc{:>5.2f}% | Time: {:>.2f}s'.format(
                        round_i, c.cid, i, self.clients_per_round,
                        stat['loss'], stat['acc'] * 100, stat['time']
                    ))

                # Add stats
                stats.append(stat)

            # Get average for global parameters
            for key in self.w_glob_keys:
                self.latest_model[key] = torch.div(self.latest_model[key], len(selected_clients))

            # Track communication cost
            non_commu_clients = [c for c in self.clients if c not in selected_clients]
            for c in non_commu_clients:
                stat = {'id': c.cid, 'bytes_w': 0, 'bytes_r': self.num_param_glob}
                stats.append(stat)
            self.metrics.extend_commu_stats(round_i, stats)

        # Test final model on train data
        self.test_latest_model_on_train_data(self.num_round)
        self.test_latest_model_on_eval_data(self.num_round)

        # Save tracked information
        self.metrics.write()

    def test_latest_model_on_train_data(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data = False)
        acc = sum(stats_from_train_data['acc']) / sum(stats_from_train_data['num_samples'])
        loss = sum(stats_from_train_data['loss']) / sum(stats_from_train_data['num_samples'])
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss:{:.4f} /'
            ' Time: {:.2f}s'.format(
                round_i, acc, loss, end_time - begin_time
            ))
            print('=' * 102 + "\n")

    def test_latest_model_on_eval_data(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data = True)
        acc = sum(stats_from_eval_data['acc']) / sum(stats_from_eval_data['num_samples'])
        loss = sum(stats_from_eval_data['loss']) / sum(stats_from_eval_data['num_samples'])
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print("= Test = round: {} / acc: {:.3%} / ".format(round_i, acc),
            "loss: {:.4f} / Time: {:.2f}s".format(loss, end_time - begin_time))
            print("=" * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data = True):
        num_samples = []
        accs = []
        losses = []
        netlosses = []

        for c in self.clients:
            local_model = self.update_local_model(c.local_model)
            c.model.load_state_dict(local_model)
            test_dict = c.local_test(use_eval_data = use_eval_data)
            num_sample = test_dict['test_num']
            acc = test_dict['acc']
            loss = test_dict['loss']
            netloss = test_dict['loss']

            num_samples.append(num_sample)
            accs.append(acc)
            losses.append(loss)
            netlosses.append(netloss)

        ids = [c.cid for c in self.clients]

        stats = {'acc': accs, 'loss': losses, 'netloss': netlosses, 'num_samples': num_samples, 'ids': ids}

        return stats