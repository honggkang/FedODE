import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import ssl
import wandb
ssl._create_default_https_context = ssl._create_unverified_context

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import time
from utilsHessian import *
from density_plot import get_esd_plot
from pyhessian import hessian

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--method', type=str, default='euler')
parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
parser.add_argument('--iid', default='False', action='store_true', help='whether i.i.d or not')
parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')
parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--verbose', action='store_true', help='verbose print')
parser.add_argument('--integration_time_num', type=int, default=9)
parser.add_argument('--dataset', default='mnist')

parser.add_argument('--save', type=str, default='./experiment-fl-resODE/mnist')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=1)

parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
# parser.add_argument('--seed',
#                     type=int,
#                     default=1,
#                     help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    # action='store_false',
                    default=True,
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')


args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.from_numpy(np.linspace(0, 1, num=args.integration_time_num)).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method=args.method)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':
    wandb.init()
    wandb.config.update(args)
    
    makedirs(args.save)
    now = time.localtime()
    exp_time = '_'+str(now.tm_year)+str(now.tm_mon)+str(now.tm_mday)+'_'+str(now.tm_hour)+str(now.tm_min)
    output_filename = 'logs' + str(args.integration_time_num) + exp_time

    logger = get_logger(logpath=os.path.join(args.save, output_filename), filepath=os.path.abspath(__file__))
    logger.info(args)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = getData(name='mnist',
                                        train_bs=args.mini_hessian_batch_size,
                                        test_bs=1)

    assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
    assert (50000 % args.hessian_batch_size == 0)
    batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

    if batch_num == 1:
        for inputs, labels in train_loader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(train_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break

    is_odenet = args.network == 'odenet'

    if args.dataset == 'mnist':
        init_ch = 1
    elif args.dataset == 'cifar':
        init_ch = 3

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(init_ch, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(init_ch, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    glob_model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    w_glob = glob_model.state_dict()
    wandb.watch(glob_model)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    logger.info(glob_model)
    logger.info('Number of parameters: {}'.format(count_parameters(glob_model)))

    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    criterion = nn.CrossEntropyLoss().to(device)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if args.data_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train) # augmentation on client should be implemented
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    # sample users
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # train_loader, test_loader, train_eval_loader = get_mnist_loaders(
    #     args.data_aug, args.batch_size, args.test_batch_size
    # )

    # data_gen = inf_generator(train_loader)
    # batches_per_epoch = len(train_loader)



    # lr_fn = learning_rate_with_decay(
    #     args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
    #     decay_rates=[1, 0.1, 0.01, 0.001]
    # )

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs):
        glob_model.train()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # if itr == 20:
        #     args.lr = args.lr/5
        #
        # if itr == 150:
        #     args.lr = args.lr/10
        #
        # if itr == 180:
        #     args.lr = args.lr/10

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(glob_model).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals)

        glob_model.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Learning rate {:4f}'.format(itr, loss_avg, args.lr))
        loss_train.append(loss_avg)

        wandb.log({
            "Round": itr,
            "Average loss": loss_avg,
        })


        # if itr<30 or (itr >= 30 and itr%3==0):
        if itr % 4 == 0:
            glob_model.eval()
            if batch_num == 1:
                hessian_comp = hessian(args.gpu,
                                       glob_model,
                                       criterion,
                                       data=hessian_dataloader,
                                       cuda=args.cuda
                                       )
            else:
                hessian_comp = hessian(args.gpu,
                                       glob_model,
                                       criterion,
                                       dataloader=hessian_dataloader,
                                       cuda=args.cuda
                                       )

            print(
                '********** finish data loading and begin Hessian computation **********')

            top_eigenvalues, _ = hessian_comp.eigenvalues()
            trace = hessian_comp.trace()
            density_eigen, density_weight = hessian_comp.density()

            logger.info('Top Eigenvalues: {}'.format(top_eigenvalues[0]))
            logger.info('Trace: {}'.format(np.mean(trace)))

            acc_train, loss_train_f = test_img(glob_model, dataset_train, args)
            acc_test, loss_test_f = test_img(glob_model, dataset_test, args)
            # print("Training accuracy: {:.2f}".format(acc_train))
            # print("Testing accuracy: {:.2f}".format(acc_test))

            # "Communication round {:04d} | Train Acc {:.4f} | Test Acc {:.4f}"
            # tt = 'model' + str(itr) + '.pth'
            # torch.save({'state_dict': glob_model.state_dict(), 'args': args}, os.path.join(args.save, tt))
            logger.info(
                            "{:04d} {:.4f} {:.4f}".format(itr, acc_train, acc_test)
            )

            wandb.log({
                "Test accuracy": acc_test,
                "Train accuracy": acc_train,
                "Top eigenvalues": top_eigenvalues[0],
                "Trace": np.mean(trace),
                "Eigen density": density_eigen,
                "Density weight": density_weight,
            })

    # for itr in range(args.nepochs * batches_per_epoch):
    #
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr_fn(itr)
    #
    #     optimizer.zero_grad()
    #     x, y = data_gen.__next__()
    #     x = x.to(device)
    #     y = y.to(device)
    #     logits = model(x)
    #     loss = criterion(logits, y)
    #
    #     if is_odenet:
    #         nfe_forward = feature_layers[0].nfe
    #         feature_layers[0].nfe = 0
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     if is_odenet:
    #         nfe_backward = feature_layers[0].nfe
    #         feature_layers[0].nfe = 0
    #
    #     batch_time_meter.update(time.time() - end)
    #     if is_odenet:
    #         f_nfe_meter.update(nfe_forward)
    #         b_nfe_meter.update(nfe_backward)
    #     end = time.time()
    #
    #     if itr % batches_per_epoch == 0:
    #         with torch.no_grad():
    #             train_acc = accuracy(model, train_eval_loader)
    #             val_acc = accuracy(model, test_loader)
    #             if val_acc > best_acc:
    #                 torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
    #                 best_acc = val_acc
    #             logger.info(
    #                 "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
    #                 "Train Acc {:.4f} | Test Acc {:.4f}".format(
    #                     itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
    #                     b_nfe_meter.avg, train_acc, val_acc
    #                 )
    #             )