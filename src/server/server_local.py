import numpy as np
import torch
import time
from src.utils.worker_utils import Metrics
from src.client.local import LocalClient

class ServerLocal(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.eval_every = options['eval_every']

        # Initialize system metrics
        self.name = '_'.join([name,  f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']

    def setup_clients(self, dataset, options):
        users, train_data, valid_data, test_data = dataset

        all_clients = []
        for user in users:
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = LocalClient(user_id, train_data[user], valid_data[user], test_data[user], options)
            all_clients.append(c)

        return all_clients

    def train(self):
        for round_i in range(self.num_round):
            print('>>> Round {}'.format(round_i))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_valid_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # Do local update for all clients
            solns = [] # Buffer for receiving client solutions
            stats = [] # Buffer for receiving client statistics
            for c in self.clients:
                # Solve local and personal minimization
                soln, stat = c.local_train()

                if self.print_result:
                    print('Round: {: >2d} | CID:{: >3d} |'
                    'personalized model param: norm {: >.4f} ({:>4f}->{:>4f})| '
                    'loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s'.format(
                        round_i, c.cid, 
                        stat['norm'], stat['min'], stat['max'],
                        stat['loss'], stat['acc'] * 100, stat['time']
                    ))
                
                # Add solutions and stats
                solns.append(soln)
                stats.append(stat)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

        # Test final model on train data
        self.test_latest_model_on_train_data(self.num_round)
        self.test_latest_model_on_valid_data(self.num_round)
        self.test_latest_model_on_eval_data(self.num_round)

        # Save tracked information
        self.metrics.write()

    def test_latest_model_on_train_data(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data = 0)
        acc = sum(stats_from_train_data['acc']) / sum(stats_from_train_data['num_samples'])
        loss = sum(stats_from_train_data['loss']) / sum(stats_from_train_data['num_samples'])
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                ' Time: {:.2f}s'.format(
                    round_i, acc, loss, end_time - begin_time
                ))
            print('=' * 102 + "\n")

    def test_latest_model_on_valid_data(self, round_i):
        # Collect stats from total valid data
        stats_from_valid_data = self.local_test(use_eval_data = 1)
        acc = sum(stats_from_valid_data['acc']) / sum(stats_from_valid_data['num_samples'])
        loss = sum(stats_from_valid_data['loss']) / sum(stats_from_valid_data['num_samples'])

        self.metrics.update_valid_stats(round_i, stats_from_valid_data)

    def test_latest_model_on_eval_data(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data = 2)
        acc = sum(stats_from_eval_data['acc']) / sum(stats_from_eval_data['num_samples'])
        loss = sum(stats_from_eval_data['loss']) / sum(stats_from_eval_data['num_samples'])
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print("= Test = round: {} / acc: {:.3%} / ".format(round_i, acc), 
            "loss: {:.4f} / Time: {:.2f}s".format(loss, end_time - begin_time))
            print("=" * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data = 2):
        num_samples = []
        accs = []
        losses = []
        netlosses = []

        for c in self.clients:
            test_dict = c.local_test(use_eval_data = use_eval_data)
            num_sample = test_dict['test_num']
            acc = test_dict['acc']
            loss = test_dict['loss']
            netloss = test_dict['netloss']

            num_samples.append(num_sample)
            accs.append(acc)
            losses.append(loss)
            netlosses.append(netloss)

        ids = [c.cid for c in self.clients]

        stats = {'acc': accs, 'loss': losses, 'netloss': netlosses, 'num_samples': num_samples, 'ids': ids}

        return stats