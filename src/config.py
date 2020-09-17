class TrainConfig:

    def __init__(self, base_path, n_epochs, batch_size, lr, dim_h, n_z, LAMBDA, 
                    sigma, n_channel, img_size):

        self.base_path = base_path   # project directory path
        self.n_epochs = n_epochs     # number of epochs to train
        self.batch_size = batch_size # input batch size for training
        self.lr = lr                 # learning rate (default: 0.0001)
        self.dim_h = dim_h           # hidden dimension (default: 128)
        self.n_z = n_z               # hidden dimension of z (default: 8)
        self.LAMBDA = LAMBDA         # regularization coef term (default: 10)
        self.sigma = sigma           # variance of hidden dimension
        self.n_channel = n_channel   # input channels
        self.img_size = img_size     # image size