class TrainConfig:

    def __init__(self, base_path, n_epochs, batch_size, lr, dim_h, n_z, LAMBDA,
                 sigma, n_channel, img_size):
        """
        base_path: project directory path
        n_epochs: number of epochs to train
        batch_size: input batch size for training
        lr : learning rate (default: 0.0001)
        dim_h : hidden dimension (default: 128)
        n_z : hidden dimension of z (default: 8)
        LAMBDA : regularization coef term (default: 10)
        sigma : variance of hidden dimension
        n_channel: input channels
        img_size: image size
        """

        self.base_path = base_path
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dim_h = dim_h
        self.n_z = n_z
        self.LAMBDA = LAMBDA
        self.sigma = sigma
        self.n_channel = n_channel
        self.img_size = img_size
