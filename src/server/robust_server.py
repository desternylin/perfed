from src.server.server import Server
import numpy as np
import torch
import time
import importlib
from src.utils.worker_utils import Metrics
from config import ALGORITHMS

class RobustServer(Server):
    def __init__(self, dataset, options, name = ''):
        super(RobustServer, self).__init__(dataset, options, name)
        self.attack = options['attack']
        self.mali_frac = options['mali_frac']
        
        # Determine the malicious clients
        self.all_clients_id = self.get_all_clients_id()
        np.random.seed(options['seed'])
        self.mali_clients = [self.all_clients_id[i] for i in np.random.choice(len(self.all_clients_id), round(len(self.all_clients_id) * self.mali_frac), replace = False)]
        self.benign_clients = list(filter(lambda i: i not in self.mali_clients, self.all_clients_id))

        if self.attack == 'data_poison':
            for mali_id in self.mali_clients:
                if 'emnist' in options['dataset']:
                    y_range = 62
                else:
                    y_range = 10
                c = self.clients[mali_id]
                self.clients[mali_id].train_data.labels = np.random.randint(0, y_range, len(c.train_data.labels))

    def get_all_clients_id(self):
        clients_id = []
        for c in self.clients:
            clients_id.append(c.cid)
        return clients_id

    def aggregate(self, solns, seed, stats):
        averaged_solution = torch.zeros_like(self.latest_model)

        # Select K clients
        if len(solns) != self.clients_per_round:
            num_clients = min(self.clients_per_round, len(self.clients))
            np.random.seed(seed)
            chosen_index = np.random.choice(len(self.all_clients_id), num_clients, replace = False)
            chosen_id = [self.all_clients_id[i] for i in chosen_index]
            solns = [solns[i] for i in chosen_index]
        else:
            chosen_id = []
            for cstat in stats:
                chosen_id.append(cstat['id'])
                
        num_samples = []
        chosen_solns = []

        for cid, (num_sample, local_soln) in zip(chosen_id, solns):
            if cid in self.mali_clients:
                if self.attack == 'same_value':
                    local_soln = torch.ones_like(self.latest_model) * np.random.normal(0, 100, 1).item()
                elif self.attack == 'sign_flip':
                    magnitude = abs(np.random.normal(0, 10, 1).item())
                    local_soln = - magnitude * local_soln
                elif self.attack == 'gaussian':
                    local_soln = torch.normal(mean = 0., std = 100., size = local_soln.size())
                    if self.gpu:
                        local_soln = local_soln.to(self.device)
                elif self.attack == 'data_poison':
                    local_soln = local_soln * np.random.normal(0, 20, 1).item()
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
            f = int(len(chosen_solns) * self.mali_frac)
            dists = torch.zeros(len(chosen_solns), len(chosen_solns))
            scores = torch.zeros(len(chosen_solns))
            for i in range(len(chosen_solns)):
                for j in range(i, len(chosen_solns)):
                    dists[i][j] = torch.norm(chosen_solns[i] - chosen_solns[j], p = 2)
                    dists[j][i] = dists[i][j]
            for i in range(len(chosen_solns)):
                d = dists[i]
                d, _ = d.sort()
                scores[i] = d[:len(chosen_solns) - f - 1].sum()
            averaged_solution = chosen_solns[torch.argmin(scores).item()]


        averaged_solution = (1 - self.beta) * self.latest_model + self.beta * averaged_solution
        return averaged_solution.detach()

    def local_test(self, use_eval_data = 2):
        assert self.latest_model is not None

        num_samples = []
        accs = []
        losses = []
        netlosses = []

        for c in self.clients:
            if c.cid in self.benign_clients:
                c.set_local_model_params(self.latest_model)

                proj_algo = ['proj', 'lp_proj', 'proj_fair']
                if self.algo in proj_algo:
                    test_dict = c.local_test(self.Proj, use_eval_data = use_eval_data)
                else:
                    test_dict = c.local_test(use_eval_data = use_eval_data)
                num_sample = test_dict['test_num']
                acc = test_dict['acc']
                loss = test_dict['loss']
                netloss = test_dict['netloss']

                num_samples.append(num_sample)
                accs.append(acc)
                losses.append(loss)
                netlosses.append(netloss)

        ids = self.benign_clients

        stats = {'acc': accs, 'loss': losses, 'netloss': netlosses, 'num_samples': num_samples, 'ids': ids}

        return stats