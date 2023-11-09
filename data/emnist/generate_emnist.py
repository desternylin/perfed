import torch
import numpy as np
import pickle
import os
import torchvision
cpath = os.path.dirname(__file__)

NUM_USER = 248
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = False
np.random.seed(2021)

class ImageDataset(object):
    def __init__(self, images, labels, normalize = False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy() / 255
            else:
                self.data = images.numpy()
        else:
            self.data = images

        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)

def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst

def choose_digit(split_data_lst, num_digit):
    available_digit = []
    available_digit_freq = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
            available_digit_freq.append(len(digit))

    max_freq_digit_ind = [v for v in range(len(available_digit)) if available_digit_freq[v] == max(available_digit_freq)]
    max_freq_digit = [available_digit[v] for v in max_freq_digit_ind]

    if len(max_freq_digit) < num_digit:
        lst = []
        lst += max_freq_digit
        second_max = sorted(available_digit_freq)[-(len(max_freq_digit)+1)]
        second_max_freq_digit_ind = [v for v in range(len(available_digit)) if available_digit_freq[v] == second_max]
        second_max_freq_digit = [available_digit[v] for v in second_max_freq_digit_ind]
        lst += np.random.choice(second_max_freq_digit, num_digit - len(max_freq_digit), replace = False).tolist()
    else:
        lst = np.random.choice(max_freq_digit, num_digit, replace = False).tolist()

    return lst

def generate_for_task(equal = True):
    # Get EMNIST data, normalize, and divide by level
    print('>>> Get EMNIST data.')
    trainset = torchvision.datasets.EMNIST(DATASET_FILE, split = 'byclass', download = True, train = True)
    testset = torchvision.datasets.EMNIST(DATASET_FILE, split = 'byclass', download = True, train = False)
    train_emnist = ImageDataset(trainset.train_data, trainset.train_labels)
    test_emnist = ImageDataset(testset.test_data, testset.test_labels)

    # Sort the data according to their classes
    emnist_traindata = []
    for number in range(62):
        idx = train_emnist.target == number
        emnist_traindata.append(train_emnist.data[idx])

    if equal:
        min_number = min([len(dig) for dig in emnist_traindata])
        # Force each digit to have the same amount of train data
        for number in range(62):
            emnist_traindata[number] = emnist_traindata[number][:min_number-1]
        
    split_emnist_traindata = []
    for digit in emnist_traindata:
        split_emnist_traindata.append(data_split(digit, int(NUM_USER * 10 / 62)))

    emnist_testdata = []
    for number in range(62):
        idx = test_emnist.target == number
        emnist_testdata.append(test_emnist.data[idx])
    split_emnist_testdata = []
    for digit in emnist_testdata:
        split_emnist_testdata.append(data_split(digit, int(NUM_USER * 10 / 62)))

    data_distribution = np.array([len(v) for v in emnist_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_emnist_traindata])
    print('>>> Each digit in train data is split into {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_emnist_testdata])
    print('>>> Each digit in test data is split into {}'.format(digit_count.tolist()))

    # Assign samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    valid_X = [[] for _ in range(NUM_USER)]
    valid_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print('>>> Data is non-i.i.d. distributed')
    print('>>> Data is {}'.format('balance' if equal else 'unbalance'))

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_emnist_traindata]))

        for d in choose_digit(split_emnist_traindata, 10):
            l = len(split_emnist_traindata[d][-1])
            X = split_emnist_traindata[d].pop().tolist()
            valid_len = int(0.2 * l)
            train_len = l - valid_len
            # train_X[user] += split_emnist_traindata[d].pop().tolist()
            # train_y[user] += (d * np.ones(l)).tolist()
            train_X[user] += X[:train_len]
            train_y[user] += (d * np.ones(train_len)).tolist()
            valid_X[user] += X[train_len:]
            valid_y[user] += (d * np.ones(valid_len)).tolist()

            l = len(split_emnist_testdata[d][-1])
            test_X[user] += split_emnist_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print('>>> Set data path for EMNIST')
    image = 1 if IMAGE_DATA else 0
    train_path = "data/train/all_data_{}_{}_niid.pkl".format(image, 'equal' if equal else 'random')
    valid_path = "data/valid/all_data_{}_{}_niid.pkl".format(image, 'equal' if equal else 'random')
    test_path = "data/test/all_data_{}_{}_niid.pkl".format(image, 'equal' if equal else 'random')

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(valid_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    valid_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup users
    for i in range(NUM_USER):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y':train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        valid_data['users'].append(uname)
        valid_data['user_data'][uname] = {'x': valid_X[i], 'y': valid_y[i]}
        valid_data['num_samples'].append(len(valid_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x':test_X[i], 'y':test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE: 
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(valid_path, 'wb') as outfile:
            pickle.dump(valid_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)

        print('>>> Save data.')

if __name__ == '__main__':
    task = {'balance': 1, 'unbalance': 0}
    for key, value in task.items():
        generate_for_task(value)