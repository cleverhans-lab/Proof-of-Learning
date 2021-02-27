import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def get_parameters(net, numpy=False):
    # get weights from a torch model as a list of numpy arrays
    parameter = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
    if numpy:
        return parameter.cpu().numpy()
    else:
        return parameter


def set_parameters(net, parameters, device):
    # load weights from a list of numpy arrays to a torch model
    for i, (name, param) in enumerate(net.named_parameters()):
        param.data = torch.Tensor(parameters[i]).to(device)
    return net


def create_sequences(batch_size, dataset_size, epochs):
    # create a sequence of data indices used for training
    sequence = np.concatenate([np.random.default_rng().choice(dataset_size, size=dataset_size, replace=False)
                               for i in range(epochs)])
    num_batch = int(len(sequence) // batch_size)
    return np.reshape(sequence[:num_batch * batch_size], [num_batch, batch_size])


def consistent_type(model, architecture=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), half=False):
    # this function takes in directory to where model is saved, model weights as a list of numpy array,
    # or a torch model and outputs model weights as a list of numpy array
    if isinstance(model, str):
        assert architecture is not None
        state = torch.load(model)
        net = architecture()
        net.load_state_dict(state['net'])
        weights = get_parameters(net)
    elif isinstance(model, np.ndarray):
        weights = torch.tensor(model)
    elif not isinstance(model, torch.Tensor):
        weights = get_parameters(model)
    else:
        weights = model
    if half:
        weights = weights.half()
    return weights.to(device)


def parameter_distance(model1, model2, order=2, architecture=None, half=False):
    # compute the difference between 2 checkpoints
    weights1 = consistent_type(model1, architecture, half=half)
    weights2 = consistent_type(model2, architecture, half=half)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        if o == 'inf':
            o = np.inf
        if o == 'cos' or o == 'cosine':
            res = (1 - torch.dot(weights1, weights2) /
                   (torch.norm(weights1) * torch.norm(weights1))).cpu().numpy()
        else:
            if o != np.inf:
                try:
                    o = int(o)
                except:
                    raise TypeError("input metric for distance is not understandable")
            res = torch.norm(weights1 - weights2, p=o).cpu().numpy()
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def load_dataset(dataset, train, download=False):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if dataset == "MNIST" or dataset == "FashionMNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
    elif dataset == "CIFAR100":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    else:
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data = dataset_class(root='./data', train=train, download=download, transform=transform)
    return data


def ks_test(reference, rvs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        ecdf = torch.arange(rvs.shape[0]).float() / torch.tensor(rvs.shape)
        return torch.max(torch.abs(reference(torch.sort(rvs)[0]).to(device) - ecdf.to(device)))


def check_weights_initialization(param, method):
    if method == 'default':
        # kaimin uniform (default for weights of nn.Conv and nn.Linear)
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std
        reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    elif method == 'resnet_cifar':
        # kaimin normal
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.normal.Normal(0, std).cdf
    elif method == 'resnet':
        # kaimin normal (default in conv layers of pytorch resnet)
        fan = nn.init._calculate_correct_fan(param, 'fan_out')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.normal.Normal(0, std).cdf
    elif method == 'default_bias':
        # default for bias of nn.Conv and nn.Linear
        weight, param = param
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in)
        reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    else:
        raise NotImplementedError("Input initialization strategy is not implemented.")

    param = param.reshape(-1)
    ks_stats = ks_test(reference, param).cpu().item()
    return stats.kstwo.sf(ks_stats, param.shape[0])


def check_weights_initialization_scipy(param, method):
    if method == 'default':
        # kaimin uniform (default for weights of nn.Conv and nn.Linear)
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std
        reference = stats.uniform(loc=-bound, scale=bound * 2).cdf
    elif method == 'resnet':
        # kaimin normal (default in conv layers of pytorch resnet)
        fan = nn.init._calculate_correct_fan(param, 'fan_out')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = stats.norm(loc=0, scale=std).cdf
    elif method == 'default_bias':
        # default for bias of nn.Conv and nn.Linear
        weight, param = param
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in)
        reference = stats.uniform(loc=-bound, scale=bound * 2).cdf
    else:
        raise NotImplementedError("Input initialization strategy is not implemented.")

    param = param.detach().numpy().reshape(-1)
    return stats.kstest(param, reference)[1]
