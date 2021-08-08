from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch
import copy
import numpy as np

class grad_desc(Optimizer):
    def __init__(self, params, lr = required, weight_decay = 0.0):
        self.lr = lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr = lr, weight_decay = weight_decay)
        super(grad_desc, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure = None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                if beta != 0:
                    p.data.add_(d_p, alpha = -beta)
                else:
                    p.data.add_(d_p, alpha = -group['lr'])

        return loss

    def adjust_learning_rate(self, round_i):
        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i + 1)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']

class me_optimizer(grad_desc):
    def __init__(self, params, lr = required, lamda = 1.0, weight_decay = 0.0):
        self.lamda = lamda        
        super(me_optimizer, self).__init__(params, lr, weight_decay)

    def step(self, local_weight, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1
        regularizer = copy.deepcopy(self.param_groups[0]['params'])
        flat_person_weight = get_flat_params_from_param_groups(regularizer)
        flat_regularizer = self.lamda * (local_weight - flat_person_weight)
        regularizer = set_flat_params_to_param_groups(regularizer, flat_regularizer)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p, weight_update in zip(group['params'], regularizer):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data = p.data - group['lr'] * (d_p - weight_update.data)

        return loss

    def get_optimizer_param(self):
        return {'lamda': self.lamda}
        
class proj_optimizer(grad_desc):
    def __init__(self, params, Proj, lr = required, lamda = 1.0, weight_decay = 0.0):
        self.lamda = lamda
        self.Proj = Proj

        super(proj_optimizer, self).__init__(params, lr, weight_decay)

    def step(self, local_weight, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        assert len(self.param_groups) == 1
        regularizer = copy.deepcopy(self.param_groups[0]['params'])
        flat_person_weight = get_flat_params_from_param_groups(regularizer)
        flat_regularizer = self.lamda * torch.matmul(self.Proj.t().data, local_weight - torch.matmul(self.Proj.data, flat_person_weight))
        regularizer = set_flat_params_to_param_groups(regularizer, flat_regularizer)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p, weight_update in zip(group['params'], regularizer):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data = p.data - group['lr'] * (d_p - weight_update.data)

        return loss

    def get_optimizer_param(self):
        return{'lamda': self.lamda, 'Proj': self.Proj}

class lp_optimizer(grad_desc):
    def __init__(self, params, p, lr = required, lamda = 1.0, weight_decay = 0.0):
        self.p = p
        self.lamda = lamda

        super(lp_optimizer, self).__init__(params, lr, weight_decay)

    def step(self, local_weight, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        assert len(self.param_groups) == 1
        regularizer = copy.deepcopy(self.param_groups[0]['params'])
        flat_person_weight = get_flat_params_from_param_groups(regularizer)
        conv_weight = local_weight - flat_person_weight
        flat_regularizer = self.lamda * torch.sign(conv_weight) * (torch.abs(conv_weight) / torch.norm(conv_weight, p = self.p))**(self.p - 1)
        regularizer = set_flat_params_to_param_groups(regularizer, flat_regularizer)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p, weight_update in zip(group['params'], regularizer):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data = p.data - group['lr'] * (d_p - weight_update.data)

        return loss

    def get_optimizer_param(self):
        return{'lamda': self.lamda, 'p': self.p}

class lp_proj_optimizer(grad_desc):
    def __init__(self, params, p, Proj, lr = required, lamda = 1.0, weight_decay = 0.0):
        self.p = p
        self.Proj = Proj
        self.lamda = lamda
        
        super(lp_proj_optimizer, self).__init__(params, lr, weight_decay)

    def step(self, local_weight, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1
        regularizer = copy.deepcopy(self.param_groups[0]['params'])
        flat_person_weight = get_flat_params_from_param_groups(regularizer)
        conv_weight = local_weight - torch.matmul(self.Proj.data, flat_person_weight)
        flat_regularizer = self.lamda * torch.matmul(self.Proj.t().data, torch.sign(conv_weight) * (torch.abs(conv_weight) / torch.norm(conv_weight, p = self.p))**(self.p - 1))
        regularizer = set_flat_params_to_param_groups(regularizer, flat_regularizer)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p, weight_update in zip(group['params'], regularizer):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add(p.data, alpha = weight_decay)
                p.data = p.data - group['lr'] * (d_p - weight_update.data)

        return loss

    def get_optimizer_param(self):
        return{'lamda': self.lamda, 'p': self.p, 'Proj': self.Proj}

class fair_optimizer(grad_desc):
    def __init__(self, params, q, lr = required, lamda = 1.0, weight_decay = 0.0):
        self.q = q
        self.lamda = lamda
        super(fair_optimizer, self).__init__(params, lr, weight_decay)

    def step(self, local_weight, cur_loss, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1
        regularizer = copy.deepcopy(self.param_groups[0]['params'])
        flat_person_weight = get_flat_params_from_param_groups(regularizer)
        flat_regularizer = self.lamda * (local_weight - flat_person_weight)
        regularizer = set_flat_params_to_param_groups(regularizer, flat_regularizer)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p, weight_update in zip(group['params'], regularizer):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data = p.data - group['lr'] * (cur_loss**self.q * d_p - weight_update.data)

        return loss

    def get_optimizer_param(self):
        return {'lamda': self.lamda, 'q': self.q}

def get_flat_params_from_param_groups(param_groups):
    params = []
    for param in param_groups:
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to_param_groups(param_groups, flat_params):
    prev_ind = 0
    for param in param_groups:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size
    return param_groups