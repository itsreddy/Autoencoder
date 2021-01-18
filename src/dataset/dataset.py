from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms


class Mnist:

    def __init__(self, args):
        self.transform = transforms.ToTensor()
        self.base_path = args.base_path
        self.batch_size = args.batch_size

    def get_data_loader(self, train: bool):
        d_loader = DataLoader(dataset=self.load_data(train),
                              batch_size=self.batch_size,
                              shuffle=True)
        return d_loader

    def load_data(self, train):
        path = self.base_path + 'data/'
        if train:
            dset = MNIST(root=path,
                         train=True,
                         transform=self.transform,
                         download=True)
        else:
            dset = MNIST(root=path,
                         train=False,
                         transform=self.transform,
                         download=True)
        return dset


class Cifar:

    def __init__(self, args):
        self.transform = transforms.ToTensor()
        self.base_path = args.base_path
        self.batch_size = args.batch_size

    def get_data_loader(self, train: bool):
        d_loader = DataLoader(dataset=self.load_data(train),
                              batch_size=self.batch_size,
                              shuffle=True)
        return d_loader

    def load_data(self, train):
        path = self.base_path + 'data/'
        if train:
            dset = CIFAR10(root=path,
                           train=True,
                           transform=self.transform,
                           download=True)
        else:
            dset = CIFAR10(root=path,
                           train=False,
                           transform=self.transform,
                           download=True)
        return dset
