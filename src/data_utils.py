from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import time, os

class CustomDataLoader():
    
    def __init__(self, args):
        self.args = args

    def load_data(self, train):
        path = self.args.base_path + 'data/'
        if train:
            dset = MNIST(root=path,
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
        else:
            dset = MNIST(root=path,
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)
        return dset
    
    def get_data_loader(self, train):
        d_loader = DataLoader(dataset=self.load_data(train),
                          batch_size=self.args.batch_size,
                          shuffle=True)
        return d_loader

class SavePath():

    def __init__(self, args, results_path=None):
        self.args = args
        if results_path:
            self.results_path = results_path
        else: 
            timeasname = time.asctime(time.localtime(time.time()))\
                            .replace(" ", "-").replace(":", "-")
            self.results_path = self.args.base_path + \
                                    "outs/{}/".format(timeasname)

    def get_save_paths(self, make_directories = True):

        image_path = self.results_path + "images/"
        list_path = self.results_path +  "lists/"
        model_path = self.results_path +  "models/"
        
        if make_directories:
            os.makedirs(image_path, exist_ok=True)
            os.makedirs(list_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)

        return image_path, list_path, model_path

    

