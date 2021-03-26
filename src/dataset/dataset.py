from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, LSUN
from torchvision.transforms import transforms

class Dataset:
    
    def __init__(self, args):
        self.transform = transforms.ToTensor()
        self.base_path = args.base_path
        self.batch_size = args.batch_size
    
    def get_data_loader(self, train):
        d_loader = DataLoader(dataset=self.load_data(train),
                              batch_size=self.batch_size,
                              shuffle=True)
    def load_data():
        pass

class Mnist(Dataset):

    def __init__(self, args):
        super().__init__(args)
        
    def get_data_loader(self, train):
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


class Cifar(Dataset):

    def __init__(self, args):
        super().__init__(args)

    def get_data_loader(self, train=True, transform=None, load_specific_classes=None):
        if transform:
            self.transform = transform
        dataset = self.load_data(train)
        if load_specific_classes:
            '''
            TODO: for some reason, the labels are not getting selected properly, but the images are getting selected from
            the specified classes
            '''
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

class Lsun(Dataset):

    def __init__(self, args):
        super().__init__(args)
    
    def get_data_loader(self, train, class_list):
        d_loader = DataLoader(dataset=self.load_data(train, class_list),
                              batch_size=self.batch_size,
                              shuffle=True)

    def load_data(self, train, class_list):
        path = self.base_path + 'data/'
        if train:
            dset = LSUN(root=path,
                         classes=class_list,
                         transform=self.transform)
        else:
            dset = LSUN(root=path,
                         classes=class_list,
                         transform=self.transform)
        return dset
