import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np

class Logistic(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(input_shape, out_dim)

        self.weight_keys = [['layer.weight', 'layer.bias']]

    def forward(self, x):
        logit = self.layer(x)
        return logit

class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim, mid_dim = 100):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, out_dim)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias']]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class OneHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim, mid_dim = 100):
        super(OneHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.weight_keys = [['conv1.weight', 'conv1.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias']]

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# VGG16 for CelebA dataset
class VGG16(nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          # (1(32-1)- 32 + 3)/2 = 1
                          padding=1), 
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
        
        self.classifier = nn.Sequential(
                nn.Linear(512*4*4, 4096),
                nn.ReLU(),   
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, num_classes)
        )
            
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()

        self.weight_keys = [['block_1.0.weight', 'block_1.0.bias'],
                            ['block_1.2.weight', 'block_1.2.bias'],
                            ['block_2.0.weight', 'block_2.0.bias'],
                            ['block_2.2.weight', 'block_2.2.bias'],
                            ['block_3.0.weight', 'block_3.0.bias'],
                            ['block_3.2.weight', 'block_3.2.bias'],
                            ['block_3.4.weight', 'block_3.4.bias'],
                            ['block_3.6.weight', 'block_3.6.bias'],
                            ['block_4.0.weight', 'block_4.0.bias'],
                            ['block_4.2.weight', 'block_4.2.bias'],
                            ['block_4.4.weight', 'block_4.4.bias'],
                            ['block_4.6.weight', 'block_4.6.bias'],
                            ['block_5.0.weight', 'block_5.0.bias'],
                            ['block_5.2.weight', 'block_5.2.bias'],
                            ['block_5.4.weight', 'block_5.4.bias'],
                            ['block_5.6.weight', 'block_5.6.bias'],
                            ['classifier.0.weight', 'classifier.0.bias'],
                            ['classifier.2.weight', 'classifier.2.bias'],
                            ['classifier.4.weight', 'classifier.4.bias']]
        
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        logits = self.classifier(x.view(-1, 512*4*4))
        return logits

# CNN for CelebA dataset
class CelebaNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CelebaNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 32,
                      kernel_size = (3, 3),
                      stride = (1, 1),
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2),
                         stride = (2, 2)),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 32,
                      kernel_size = (3, 3),
                      stride = (1, 1),
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2),
                         stride = (2, 2)),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 32,
                      kernel_size = (3, 3),
                      stride = (1, 1),
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2),
                         stride = (2, 2)),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 32,
                      kernel_size = (3, 3),
                      stride = (1, 1),
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2),
                         stride = (2, 2)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()
        
        self.weight_keys = [['block_1.0.weight', 'block_1.0.bias'], 
                            ['block_1.1.weight', 'block_1.1.bias', 'block_1.1.running_mean', 'block_1.1.running_var', 'block_1.1.num_batches_tracked'], 
                            ['block_2.0.weight', 'block_2.0.bias'], 
                            ['block_2.1.weight', 'block_2.1.bias', 'block_2.1.running_mean', 'block_2.1.running_var', 'block_2.1.num_batches_tracked'], 
                            ['block_3.0.weight', 'block_3.0.bias'], 
                            ['block_3.1.weight', 'block_3.1.bias', 'block_3.1.running_mean', 'block_3.1.running_var', 'block_3.1.num_batches_tracked'], 
                            ['block_4.0.weight', 'block_4.0.bias'], 
                            ['block_4.1.weight', 'block_4.1.bias', 'block_4.1.running_mean', 'block_4.1.running_var', 'block_4.1.num_batches_tracked'], 
                            ['classifier.0.weight', 'classifier.0.bias'], 
                            ['classifier.2.weight', 'classifier.2.bias']]

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        logits = self.classifier(x.view(-1, 32 * 8 * 8))
        return logits

class ProjectionWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, Proj, gpu = False, device = 0):
        super(ProjectionWrap, self).__init__()

        self.m = [module]

        self.name_base_localname = []

        self.initial_value = dict()

        self.full_dimension = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.intrinsic_dimension = intrinsic_dimension

        self.Proj = Proj.t()

        self.random_matrix = dict()

        if gpu:
            theta = nn.Parameter(torch.zeros((self.intrinsic_dimension, 1)).to(device))
        else:
            theta = nn.Parameter(torch.zeros((self.intrinsic_dimension, 1)))
        self.register_parameter("theta", theta)
        theta_size = (self.intrinsic_dimension,)

        prev_ind = 0
        for name, param in module.named_parameters():
            if param.requires_grad:
                if gpu:
                    self.initial_value[name] = v0 = (
                        param.clone().detach().requires_grad_(False).to(device)
                    )
                else:
                    self.initial_value[name] = v0 = (
                        param.clone().detach().requires_grad_(False)
                    )

                matrix_size = v0.size() + theta_size
                flat_size = int(np.prod(list(param.size())))
                if len(list(param.size())) == 1:
                    self.random_matrix[name] = self.Proj[prev_ind:prev_ind + flat_size, :].view(param.size()[0], self.intrinsic_dimension).to(device)
                elif len(list(param.size())) == 2:
                    self.random_matrix[name] = self.Proj[prev_ind:prev_ind + flat_size, :].view(param.size()[0], param.size()[1], self.intrinsic_dimension).to(device)
                else:
                    self.random_matrix[name] = self.Proj[prev_ind:prev_ind + flat_size, :].view(param.size()[0], param.size()[1], param.size()[2], self.intrinsic_dimension).to(device)
                prev_ind += flat_size

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

        del(self.Proj)

    def forward(self, x):
        for name, base, localname in self.name_base_localname:
            ray = torch.matmul(self.random_matrix[name], self.theta)

            param = self.initial_value[name] + torch.squeeze(ray, -1)

            setattr(base, localname, param)

        module = self.m[0]
        x = module(x)
        return x

# ResNet for FashionMNIST
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv = False, stride = 1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        
        return F.relu(y + x)

def resnet_block(in_channels, out_channels, num_residuals, first_block = False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv = True, stride = 2))
        else:
            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.block2 = nn.Sequential(
            resnet_block(64, 64, 2, first_block = True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2)
        )
        self.fc = nn.Linear(512, num_classes)

        self.weight_keys = [['block1.0.weight', 'block1.0.bias'], 
                            ['block1.1.weight', 'block1.1.bias'], 
                            ['block2.0.0.conv1.weight', 'block2.0.0.conv1.bias'], 
                            ['block2.0.0.conv2.weight', 'block2.0.0.conv2.bias'], 
                            ['block2.0.0.bn1.weight', 'block2.0.0.bn1.bias'], 
                            ['block2.0.0.bn2.weight', 'block2.0.0.bn2.bias'], 
                            ['block2.0.1.conv1.weight', 'block2.0.1.conv1.bias'], 
                            ['block2.0.1.conv2.weight', 'block2.0.1.conv2.bias'], 
                            ['block2.0.1.bn1.weight', 'block2.0.1.bn1.bias'], 
                            ['block2.0.1.bn2.weight', 'block2.0.1.bn2.bias'], 
                            ['block2.1.0.conv1.weight', 'block2.1.0.conv1.bias'], 
                            ['block2.1.0.conv2.weight', 'block2.1.0.conv2.bias'], 
                            ['block2.1.0.conv3.weight', 'block2.1.0.conv3.bias'], 
                            ['block2.1.0.bn1.weight', 'block2.1.0.bn1.bias'], 
                            ['block2.1.0.bn2.weight', 'block2.1.0.bn2.bias'], 
                            ['block2.1.1.conv1.weight', 'block2.1.1.conv1.bias'], 
                            ['block2.1.1.conv2.weight', 'block2.1.1.conv2.bias'], 
                            ['block2.1.1.bn1.weight', 'block2.1.1.bn1.bias'], 
                            ['block2.1.1.bn2.weight', 'block2.1.1.bn2.bias'], 
                            ['block2.2.0.conv1.weight', 'block2.2.0.conv1.bias'], 
                            ['block2.2.0.conv2.weight', 'block2.2.0.conv2.bias'], 
                            ['block2.2.0.conv3.weight', 'block2.2.0.conv3.bias'], 
                            ['block2.2.0.bn1.weight', 'block2.2.0.bn1.bias'], 
                            ['block2.2.0.bn2.weight', 'block2.2.0.bn2.bias'], 
                            ['block2.2.1.conv1.weight', 'block2.2.1.conv1.bias'], 
                            ['block2.2.1.conv2.weight', 'block2.2.1.conv2.bias'], 
                            ['block2.2.1.bn1.weight', 'block2.2.1.bn1.bias'], 
                            ['block2.2.1.bn2.weight', 'block2.2.1.bn2.bias'], 
                            ['block2.3.0.conv1.weight', 'block2.3.0.conv1.bias'], 
                            ['block2.3.0.conv2.weight', 'block2.3.0.conv2.bias'], 
                            ['block2.3.0.conv3.weight', 'block2.3.0.conv3.bias'], 
                            ['block2.3.0.bn1.weight', 'block2.3.0.bn1.bias'], 
                            ['block2.3.0.bn2.weight', 'block2.3.0.bn2.bias'], 
                            ['block2.3.1.conv1.weight', 'block2.3.1.conv1.bias'], 
                            ['block2.3.1.conv2.weight', 'block2.3.1.conv2.bias'], 
                            ['block2.3.1.bn1.weight', 'block2.3.1.bn1.bias'], 
                            ['block2.3.1.bn2.weight', 'block2.3.1.bn2.bias'], 
                            ['fc.weight', 'fc.bias']]

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = F.avg_pool2d(y, kernel_size = y.size()[2:])
        y = self.fc(y.view(-1, 512))
        return y

class ResNet2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.block2 = nn.Sequential(
            resnet_block(64, 64, 2, first_block = True),
            resnet_block(64, 128, 2)
        )
        self.fc = nn.Linear(128, num_classes)

        self.weight_keys = [['block1.0.weight', 'block1.0.bias'], 
                            ['block1.1.weight', 'block1.1.bias'], 
                            ['block2.0.0.conv1.weight', 'block2.0.0.conv1.bias'], 
                            ['block2.0.0.conv2.weight', 'block2.0.0.conv2.bias'], 
                            ['block2.0.0.bn1.weight', 'block2.0.0.bn1.bias'], 
                            ['block2.0.0.bn2.weight', 'block2.0.0.bn2.bias'], 
                            ['block2.0.1.conv1.weight', 'block2.0.1.conv1.bias'], 
                            ['block2.0.1.conv2.weight', 'block2.0.1.conv2.bias'], 
                            ['block2.0.1.bn1.weight', 'block2.0.1.bn1.bias'], 
                            ['block2.0.1.bn2.weight', 'block2.0.1.bn2.bias'], 
                            ['block2.1.0.conv1.weight', 'block2.1.0.conv1.bias'], 
                            ['block2.1.0.conv2.weight', 'block2.1.0.conv2.bias'], 
                            ['block2.1.0.conv3.weight', 'block2.1.0.conv3.bias'], 
                            ['block2.1.0.bn1.weight', 'block2.1.0.bn1.bias'], 
                            ['block2.1.0.bn2.weight', 'block2.1.0.bn2.bias'], 
                            ['block2.1.1.conv1.weight', 'block2.1.1.conv1.bias'], 
                            ['block2.1.1.conv2.weight', 'block2.1.1.conv2.bias'], 
                            ['block2.1.1.bn1.weight', 'block2.1.1.bn1.bias'], 
                            ['block2.1.1.bn2.weight', 'block2.1.1.bn2.bias'], 
                            ['fc.weight', 'fc.bias']]

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = F.avg_pool2d(y, kernel_size = y.size()[2:])
        y = self.fc(y.view(-1, 128))
        return y

def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'logistic':
        return Logistic(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == '1nn':
        return OneHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == 'cifar':
        return CifarNet()
    elif model_name == 'vgg':
        return VGG16(options['input_shape'], options['num_class'])
    elif model_name == 'celebacnn':
        return CelebaNet(options['input_shape'], options['num_class'])
    elif model_name == 'resnet':
        return ResNet(1, options['num_class'])
    elif model_name == 'resnet2':
        return ResNet2(1, options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))

def choose_local_model(local_model_dim):
    return torch.zeros(local_model_dim)