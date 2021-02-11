import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.dim_h),
            nn.Conv2d(self.dim_h, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 8, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(self.dim_h, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 1 * 1),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 1, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h * 8, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 1, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 1, self.n_channel, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 1, 1, 1)
        x = self.main(x)
        return x
    
class GanDiscriminator2(nn.Module):
    def __init__(self, args):
        super(GanDiscriminator2, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False), # 3 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.dim_h),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False), # 128 -> 256
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h // 2, 4, 2, 1, bias=False), # 256 -> 128
            nn.BatchNorm2d(self.dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1), # 128 -> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class GanDiscriminator(nn.Module):
    def __init__(self, args):
        super(GanDiscriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.dim_h),
            nn.Conv2d(self.dim_h, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 8, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.dim_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
class LinearEncoder(nn.Module):

    def __init__(self, args):
        super(LinearEncoder, self).__init__()
        
        self.n_z = args.n_z
        self.dim_h = args.dim_h
        self.dim_input = args.img_size ** 2

        self.main = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_h * 16),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 16, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 2),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 2, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.n_z)
        )
    
    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        return x
    

class LinearDecoder(nn.Module):

    def __init__(self, args):
        super(LinearDecoder, self).__init__()

        self.n_z = args.n_z
        self.dim_h = args.dim_h
        self.dim_output = args.img_size ** 2

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h * 2),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 2, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 8),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 8, self.dim_h * 16),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 16, self.dim_output),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.main(x)
        return x

    
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x
