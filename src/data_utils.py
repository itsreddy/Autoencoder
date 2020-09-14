from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class CustomDataLoader():
    
    def __init__(self, args):
        self.args = args

    def load_data(self, train):
        if train:
            dset = MNIST(root='./data/',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
        else:
            dset = MNIST(root='./data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)
        return dset
    
    def get_data_loader(self, train):
        d_loader = DataLoader(dataset=load_data(train),
                          batch_size=self.args.batch_size,
                          shuffle=True)
        return d_loader