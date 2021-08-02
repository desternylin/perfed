import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math

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
        # self.fc1 = nn.Linear(input_shape, out_dim)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = self.fc1(x)
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
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
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
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
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

def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'logistic':
        return Logistic(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == '1nn':
        return OneHiddenLayerFc(options['input_shape'], options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))

def choose_local_model(local_model_dim):
    return torch.zeros(local_model_dim)