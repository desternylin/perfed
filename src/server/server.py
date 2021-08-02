import numpy as np
import torch
import time
import importlib
from src.utils.worker_utils import Metrics
from config import ALGORITHMS

class Server(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset, options)
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
        self.latest_model = torch.zeros(options['local_model_dim'])
        self.aggr = options['aggr']

    def setup_clients(self, dataset, options):
        users, train_data, test_data = dataset

        # Load selected client
        client_path = 'src.client.%s' % options['algo']
        mod = importlib.import_module(client_path)
        client_class = getattr(mod, ALGORITHMS[options['algo']])

        all_clients = []
        for user in users:
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = client_class(user_id, train_data[user], test_data[user], options)
            all_clients.append(c)
        
        return all_clients

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        for round_i in range(self.num_round):
            print('>>> Round {}, latest model.norm = {}'.format(round_i, self.latest_model.norm()))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # Do local update for all clients
            solns = [] # Buffer for receiving client solutions
            stats = [] # Buffer for receiving client statistics
            for c in self.clients:
                # Communicate the latest global model
                c.set_local_model_params(self.latest_model)

                # Solve local and personal minimization
                soln, stat = c.local_train()

                # Decay the learning rate for both local optimization problem
                c.inverse_prop_decay_learning_rate(round_i)

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

            # Choose K clients and use their local model to update the global model
            self.latest_model = self.aggregate(solns, seed = round_i)

        # Test final model on train data
        self.test_latest_model_on_train_data(self.num_round)
        self.test_latest_model_on_eval_data(self.num_round)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns, seed):
        averaged_solution = torch.zeros_like(self.latest_model)

        # Select K clients
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        solns = [solns[i] for i in np.random.choice(len(solns), num_clients, replace = False)]
        chosen_solns = []
        num_samples = []
        for num_sample, local_soln in solns:
            chosen_solns.append(local_soln)
            num_samples.append(num_sample)

        if self.aggr == 'mean':        
            if self.simple_average:
                num = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    num += 1
                    averaged_solution += local_soln
                averaged_solution /= num
            else:
                selected_sample = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    averaged_solution += num_sample * local_soln
                    selected_sample += num_sample
                averaged_solution /= selected_sample
        elif self.aggr == 'median':
            stack_solution = torch.stack(chosen_solns)
            averaged_solution = torch.median(stack_solution, dim = 0)[0]
        elif self.aggr == 'krum':
            dists = torch.zeros(len(chosen_solns), len(chosen_solns))
            for i in range(len(chosen_solns)):
                for j in range(i, len(chosen_solns)):
                    dists[i][j] = torch.norm(chosen_solns[i] - chosen_solns[j], p = 2)
                    dists[j][i] = dists[i][j]
            scores = torch.sum(dists, dim = 0)
            averaged_solution = chosen_solns[torch.argmax(scores).item()]
                
        return averaged_solution.detach()

    def test_latest_model_on_train_data(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data = False)
        acc = sum(stats_from_train_data['acc']) / sum(stats_from_train_data['num_samples'])
        loss = sum(stats_from_train_data['loss']) / sum(stats_from_train_data['num_samples'])

        # Record the global gradient
        # model_len = len(self.latest_model)
        # global_grads = np.zeros(model_len)
        # num_samples = []
        # local_grads = []

        # for c in self.clients:
        #     (num, client_grad), stat = c.solve_grad()
        #     client_grad = client_grad
        #     local_grads.append(client_grad)
        #     num_samples.append(num)
        #     global_grads += client_grad * num
        # global_grads /= np.sum(np.asarray(num_samples))
        # stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # # Measure the gradient difference
        # difference = 0.
        # for idx in range(len(self.clients)):
        #     difference += np.sum(np.square(global_grads - local_grads[idx]))
        # difference /= len(self.clients)
        # stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            # print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
            #     ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
            #         round_i, acc, loss,
            #         stats_from_train_data['gradnorm'], difference, end_time - begin_time
            #     ))
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
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
        assert self.latest_model is not None

        num_samples = []
        accs = []
        losses = []
        netlosses = []

        for c in self.clients:
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