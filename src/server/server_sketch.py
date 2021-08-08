import numpy as np
import torch
import time
import importlib
from src.utils.worker_utils import Metrics
from src.client.sketch import SketchClient
from src.model.model import choose_model
from config import ALGORITHMS
from csvec import CSVec
from src.utils.worker_utils import topk

class ServerSketch(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.all_train_data_num = 0
        self.model = choose_model(options)
        self.local_lr = options['local_lr']
        self.latest_model = self.get_flat_model_params()
        # if self.gpu:
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"
        
        self.num_round = options['num_round']
        self.eval_every = options['eval_every']

        # Issues on sketching
        self.SketchParamsLargerThan = options['sketchparamslargerthan']
        self.SketchBiases = options['sketchbiases']
        for p in self.model.parameters():
            p.do_sketching = p.numel() >= self.SketchParamsLargerThan
        # override bias terms with whatever sketchBiases is
        for m in self.model.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.do_sketching = self.SketchBiases

        D = 0
        sketchMask = []
        for p in self.model.parameters():
            size = np.prod(p.data.shape)
            if p.do_sketching:
                sketchMask.append(torch.ones(size))
            else:
                sketchMask.append(torch.zeros(size))
            D += size
        self.D = D
        # self.sketchMask = torch.cat(sketchMask).bool().to(self.device)
        self.sketchMask = torch.cat(sketchMask).bool()
        print('D: {}'.format(self.D))
        print('sketchMask.sum(): {}'.format(self.sketchMask.sum()))

        self.full_model_dim = options['person_model_dim']
        if options['c'] < 1:
            self.sketch_c = round(self.full_model_dim * options['c'])
        else:
            self.sketch_c = options['c']
        if options['r'] < 1:
            self.sketch_r = round(self.full_model_dim * options['r'])
        else:
            self.sketch_r = options['r']
        if options['k'] < 1:
            self.sketch_k = round(self.full_model_dim * options['k'])
        else:
            self.sketch_k = options['k']

        if self.sketch_k * options['p2'] < self.full_model_dim:
            self.p2 = options['p2']
        else:
            self.p2 = int(self.full_model_dim / self.sketch_k)
        self.doAccumulateError = options['doaccumulateerror']

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
            options['D'] = self.D
            options['sketchMask'] = self.sketchMask
            c = SketchClient(user_id, train_data[user], test_data[user], options, self.model)
            all_clients.append(c)

        return all_clients

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

    def step(self, weightUpdate):
        flat_model_params = self.get_flat_model_params()
        new_model_params = flat_model_params - self.local_lr * weightUpdate
        self.set_flat_params_to(new_model_params)

    def select_clients(self, seed = 1):
        """Selects num_clients clients from possible clients"""
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace = False).tolist()

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        for round_i in range(self.num_round):
            print('>>> Round {}'.format(round_i))

            # Test latest model on train and eval data
            self.test_latest_model_on_train_data(round_i)
            self.test_latest_model_on_eval_data(round_i)

            # choose K clients for the aggregation
            selected_clients = self.select_clients(seed = round_i)

            # Compute the gradients on the selected clients locally
            workersketches = [] # Buffer for receiving client sketches
            stats = [] # Buffer for receiving client statistics
            for i, c in enumerate(selected_clients, start = 1):
                # Communicate the latest model
                self.latest_model = self.get_flat_model_params()
                c.set_flat_params_to(self.latest_model)

                # locally compute the gradient and do the sketching
                sketch, stat = c.local_train()

                if self.print_result:
                    print('Round: {: >2d} | CID:{: >3d}({:>2d}/{:>2d})|'
                    'loss {:>.4f} | Acc{:>5.2f}% | Time: {:>.2f}s'.format(
                        round_i, c.cid, i, self.clients_per_round,
                        stat['loss'], stat['acc'] * 100, stat['time']
                    ))

                # Add solutions and stats
                workersketches.append(sketch)
                stats.append(stat)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            summedSketch = np.sum(workersketches) / len(selected_clients)
            if self.p2 > 0:
                candidateTopk = summedSketch.unSketch(k = int(self.sketch_k * self.p2))
                # Get coordinates that were populated by the unsketch
                candidateHHCoords = candidateTopk.nonzero()
                candidateTopk = torch.zeros_like(candidateTopk)
                
                # Get exact values for candidateHHCoords
                for c in selected_clients:
                    toTransfer = c.v[candidateHHCoords]
                    candidateTopk[candidateHHCoords] += toTransfer
                w = topk(candidateTopk, k = self.sketch_k)
                w = w / len(selected_clients)
                weightUpdate = torch.zeros(self.D)
                weightUpdate[self.sketchMask] = w
            else:
                assert(self.p2 == 0)
                w = summedSketch.unSketch(k = self.sketch_k)
                weightUpdate = torch.zeros(self.D)
                weightUpdate[self.sketchMask] = w

            # zero out the coords of u, v that are being updated
            for c in selected_clients:
                c.u[weightUpdate.nonzero()] = 0
                c.v[weightUpdate.nonzero()] = 0

            # now deal with the non-compressed coordinates
            vs = []
            for c in selected_clients:
                vs.append(c.v[~self.sketchMask])
            weightUpdate[~self.sketchMask] = torch.sum(torch.stack(vs), dim = 0)
            for c in selected_clients:
                c.v[~self.sketchMask] = 0

            # reset the error accumulation vector every time if error accumulation is turned off
            if not self.doAccumulateError:
                for c in self.clients:
                    c.v.zero_()
            
            # do one-step update using the weightUpdate
            self.step(weightUpdate)
        
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
            self.latest_model = self.get_flat_model_params()
            c.set_flat_params_to(self.latest_model)
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