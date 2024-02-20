import numpy as np
import torch
import time
from src.utils.worker_utils import Metrics
from src.client.lbgm import LBGMClient

class ServerLBGM(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.device = options['device']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset, options)
        self.lr = options['local_lr']
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = options['simpleaverage']
        print('>>> Weight updates by {}'.format(
            'simple average' if self.simple_average else 'sample size'
        ))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
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
            c = LBGMClient(user_id, train_data[user], valid_data[user], test_data[user], options)
            all_clients.append(c)

        return all_clients

    def select_clients(self, seed = 1):
        """Selects num_clients clients from possible clients"""
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace = False).tolist()

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))
        tmp_latest_model = self.clients[0].get_flat_model_params().detach()
        self.latest_model = tmp_latest_model
        if self.gpu:
            # self.latest_model = self.latest_model.cuda()
            self.latest_model = self.latest_model.to(self.device)

        for round_i in range(self.num_round):
            print('>>> Round {}, latest model.norm = {}'.format(round_i, self.latest_model.norm()))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_valid_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # choose K clients for the aggregation
            selected_clients = self.select_clients(seed = round_i)

            # Do local update for the selected clients
            solns = []
            stats = []
            for i, c in enumerate(selected_clients, start = 1):
                # Communicate the latest global model
                c.set_local_model_params(self.latest_model)

                # Solve local and personal minimization
                soln, stat = c.local_train()

                if self.print_result:
                    print('Round: {: >2d} | CID:{: >3d}({:>2d}/{:>2d})|'
                    'loss {:>.4f} | Acc{:>5.2f}% | Time: {:>.2f}s'.format(
                        round_i, c.cid, i, self.clients_per_round,
                        stat['loss'], stat['acc'] * 100, stat['time']
                    ))

                # Add solutions and stats
                solns.append(soln)
                stats.append(stat)

            self.metrics.extend_commu_stats(round_i, stats)

            latest_grad = self.aggregate_grad(solns)
            self.latest_model -= self.lr * latest_grad
            # self.latest_model = self.aggregate(solns, seed = round_i, stats = stats)

        # Test final model on train data
        self.test_latest_model_on_train_data(self.num_round)
        self.test_latest_model_on_valid_data(self.num_round)
        self.test_latest_model_on_eval_data(self.num_round)

        # Save tracked information
        self.metrics.write()

    def aggregate_grad(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        chosen_solns = []
        num_samples = []
        for num_sample, local_grad in solns:
            chosen_solns.append(local_grad)
            num_samples.append(num_sample)

        if self.simple_average:
            num = 0
            for num_sample, local_grad in zip(num_samples, chosen_solns):
                num += 1
                local_grad = local_grad.to(self.device)
                averaged_solution += local_grad
            averaged_solution /= num
        else:
            selected_sample = 0
            for num_sample, local_grad in zip(num_samples, chosen_solns):
                averaged_solution += num_sample * local_grad
                selected_sample += num_sample
            averaged_solution /= selected_sample

        return averaged_solution.detach()


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
        assert self.latest_model is not None

        num_samples = []
        accs = []
        losses = []
        netlosses = []

        for c in self.clients:
            # self.latest_model = self.latest_model.to(c.device)
            c.set_local_model_params(self.latest_model)

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