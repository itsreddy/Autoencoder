from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
import time, os

class LoadDataset():
    
    def __init__(self, args, data: str, transform=None):
        self.args = args
        self.transform = transform
        self.data = data

    def load_data(self, train):
        path = self.args.base_path + 'data/'
        if self.transform is None:
            self.transform = transforms.ToTensor()
        if self.data == 'mnist':
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
        if self.data == 'cifar':
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
    
    def get_data_loader(self, train: bool):
        d_loader = DataLoader(dataset=self.load_data(train),
                          batch_size=self.args.batch_size,
                          shuffle=True)
        return d_loader

class SavePath():

    def __init__(self, args, checkpoint_path=None):
        self.args = args
        if checkpoint_path:
            self.results_path = checkpoint_path
        else: 
            timeasname = time.asctime(time.localtime(time.time()))\
                            .replace(" ", "-").replace(":", "-")
            self.results_path = self.args.base_path + \
                                    "outs/{}/".format(timeasname)
        
        print(self.results_path)

    def get_save_paths(self, make_directories=True):

        image_path = self.results_path + "images/"
        list_path = self.results_path +  "lists/"
        model_path = self.results_path +  "models/"
        
        if make_directories:
            os.makedirs(image_path, exist_ok=True)
            os.makedirs(list_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)

        return image_path, list_path, model_path

    

