# Global parameters
DATASETS = ['mnist', 'synthetic', 'emnist', 'fashionmnist', 'cifar', 'celeba']
MODELS = ['logistic', '2nn', '1nn', 'cifar', 'vgg']
ALGORITHMS = {'me': 'MeClient', 
                'me_fair': 'MeFairClient',
                'proj': 'ProjClient',
                'proj_fair': 'ProjFairClient',
                'ditto': 'DittoClient',
                'lp': 'LpClient',
                'lp_proj': 'LpProjClient',
                'me_fair2': 'MeFair2Client',
                'sketch': 'SketchClient',
                'lg': 'LgClient',
                'fedavg': 'FedAvgClient',
                'perfedavg': 'PerFedAvgClient',
                'local': 'LocalClient'}
OPTIMIZERS = ALGORITHMS.keys()

class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset in ['mnist', 'fashionmnist'] :
            if model == 'logistic' or model == '2nn' or model == '1nn':
                return {'input_shape': 784, 'num_class': 10}
            else:
                return {'input_shape': (28, 28, 1), 'num_class': 10}
        elif dataset == 'emnist':
            if model in ['logistic', '2nn', '1nn']:
                return {'input_shape': 784, 'num_class': 62}
            else:
                return {'input_shape': (28, 28, 1), 'num_class': 62}
        elif dataset == 'synthetic':
            return {'input_shape': 60, 'num_class': 10}
        elif dataset == 'cifar':
            return {'input_shape': (32, 32, 3), 'num_class': 10}
        elif dataset == 'celeba':
            return {'input_shape': (128, 128, 3), 'num_class': 2}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

MODEL_PARAMS = ModelConfig()

CRITERIA = ['celoss', 'mseloss']

ATTACKS = ['same_value', 'sign_flip', 'gaussian']

SERVERTYPE = {'server': 'Server',
                'robust_server': 'RobustServer',
                'server_sketch': 'ServerSketch',
                'server_lg': 'ServerLg',
                'server_local': 'ServerLocal'}
SERVERS = SERVERTYPE.keys()

AGGR = ['mean', 'median', 'krum']