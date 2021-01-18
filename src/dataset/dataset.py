from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms


class Mnist:

    def __init__(self, args):
        self.transform = transforms.ToTensor()
        self.base_path = args.base_path
        self.batch_size = args.batch_size

    def get_data_loader(self, train=True):
        d_loader = DataLoader(dataset=self.load_data(train),
                              batch_size=self.batch_size,
                              shuffle=True)
        return d_loader

    def load_data(self, train):
        path = self.base_path + 'data/'
        if train:
            dset = MNIST(root=path,
                         train=train,
                         transform=self.transform,
                         download=True)
        else:
            dset = MNIST(root=path,
                         train=train,
                         transform=self.transform,
                         download=True)
        return dset


class Cifar:

    def __init__(self, args):
        self.transform = transforms.ToTensor()
        self.base_path = args.base_path
        self.batch_size = args.batch_size

    def get_data_loader(self, train=True, load_specific_classes=None):
        dataset = self.load_data(train)

        if load_specific_classes:
            x = lambda i: True if i in set(load_specific_classes) else False
            idx = [x(i) for i in dataset.targets]
            dataset.data = dataset.data[idx]
        
        d_loader = DataLoader(dataset=dataset,
                              batch_size=self.batch_size,
                              shuffle=True)
        return d_loader

    def load_data(self, train):
        path = self.base_path + 'data/'
        if train:
            dset = CIFAR10(root=path,
                           train=train,
                           transform=self.transform,
                           download=True)
        else:
            dset = CIFAR10(root=path,
                           train=train,
                           transform=self.transform,
                           download=True)
        return dset
