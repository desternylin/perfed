import numpy as np
import torch
import time
from src.utils.worker_utils import Metrics
from src.server.server import Server
from src.model.model import choose_model
from src.optimizer.fedoptimizer import grad_desc
import importlib
from config import ALGORITHMS

class ServerGradient(Server):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu'] if 'gpu' in options else False
        self.device = options['device']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.eval_every = options['eval_every']
        self.clients_per_round = options['clients_per_round']
        self.simple_average = options['simpleaverage']

        # Initialize system metrics
        self.name = '_'.join([name, f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']

        self.model = choose_model(options)
        self.move_model_to_gpu(self.model, options)
        self.latest_model = self.get_flat_model_params()
        self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])
        self.local_lr = options['local_lr']
        self.algo = options['algo']
        self.ini_sparse_level = options['sparse_level']
        self.rising_level = options['rising_level']

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

    def setup_clients(self, dataset, options):
        users, train_data, valid_data, test_data = dataset

        print('Number of users = {}'.format(len(users)))

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
            c = client_class(user_id, train_data[user], valid_data[user], test_data[user], options)
            all_clients.append(c)

        return all_clients

    def get_flat_model_params(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def server_step(self, flat_grads, round_i):
        self.model.train()
        self.optimizer.zero_grad()
        prev_ind = 0
        for param in self.model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.grad = flat_grads[prev_ind:prev_ind + flat_size].view(param.size())
            prev_ind += flat_size
        if self.algo == 'qsgd':
            self.inverse_prop_decay_learning_rate(round_i)
        self.optimizer.step()

    def inverse_prop_decay_learning_rate(self, round_i):
        self.local_lr /= round_i + 1

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        if self.gpu:
            self.latest_model = self.latest_model.cuda()

        for round_i in range(self.num_round):
            print('>>> Round {}, latest model.norm = {}'.format(round_i, self.latest_model.norm()))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_valid_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # set sparse level for clients when dgc is applied
            if self.algo == 'dgc':
                tmp_sparse_level = 1 - (1 - self.ini_sparse_level) * pow(self.rising_level, round_i)
                cur_sparse_level = min(tmp_sparse_level, 0.999)
                self.set_client_sparse_level(cur_sparse_level)

            # choose K clients for the aggregation
            selected_clients = self.select_clients(seed = round_i)

            # compute local gradient and quantize the result
            solns = []
            stats = []
            for i, c in enumerate(selected_clients, start = 1):
                # Communicate the latest global model

                c.set_local_model_params(self.latest_model)

                # Solve local gradient
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

            # for client in range(len(stats)):
            #     print('client is {}'.format(client))
            #     stats[client]['bytes_w'] = stats[client]['bytes_w'] * (self.clients_per_round - 1)
            #     stats[client]['bytes_r'] = stats[client]['bytes_r'] * (self.clients_per_round - 1)
            #     print('client[bytes_w]= {}'.format(stats[client]['bytes_w']))

            self.metrics.extend_commu_stats(round_i, stats)

            # print(solns)

            self.aggregate(solns, round_i)
            self.latest_model = self.get_flat_model_params()

        # Test final model
        self.test_latest_model_on_train_data(self.num_round)
        self.test_latest_model_on_valid_data(self.num_round)
        self.test_latest_model_on_eval_data(self.num_round)

        # Save tracked information
        self.metrics.write()

    def set_client_sparse_level(self, cur_sparse_level):
        for c in self.clients:
            c.sparse_level = cur_sparse_level

    def aggregate(self, solns, round_i):
        averaged_grad = torch.zeros_like(self.latest_model)

        chosen_solns = []
        num_samples = []
        for num_sample, local_soln in solns:
            chosen_solns.append(local_soln)
            num_samples.append(num_sample)

        if self.simple_average:
            num = 0
            for local_soln in chosen_solns:
                num += 1
                if self.algo == 'dgc':
                    local_soln = local_soln.to(self.device)
                averaged_grad += local_soln
            averaged_grad /= num
        else:
            selected_sample = 0
            for num_sample, local_soln in zip(num_samples, chosen_solns):
                averaged_grad += num_sample * local_soln
                selected_sample += num_sample
            averaged_grad /= selected_sample

        self.server_step(averaged_grad, round_i)

    def local_test(self, use_eval_data = 2):
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

            stats = {'acc':accs, 'loss':losses, 'netloss': netlosses, 'num_samples': num_samples, 'ids': ids}

            return stats

