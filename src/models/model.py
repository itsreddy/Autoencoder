import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm_notebook as tq
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR


class AutoEncoderGAN:

    def __init__(self, args, criterion, cuda, train_loader, test_loader,
                 optimizer="Adam", checkpoint=None, load_path=None):

        """
        Creates an Autoencoder-GAN object containing the Encoder, Decoder and 
        Discriminator and also provides methods to train it.

        Input: 
            args: TrainConfig object
            criterion: loss function
            cuda: boolean if cuda is available
            checkpoint: int epoch (to load models)
            load_path: SavePath object to get file paths

        TO-DO: 
            Enable a logging interface like tensorboardX
            Implement a method to generate random images from latent space 
            Implement a method to print/save model architectures
        """

        self.reconstr_loss_epoch = []
        self.args = args
        self.criterion = criterion

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)
        self.discriminator = Discriminator(self.args)

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        if cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.discriminator = self.discriminator.cuda()

        if checkpoint:
            self.checkpoint = checkpoint
            if load_path:
                _, list_path, model_path = load_path.get_save_paths()
            else:
                raise Exception("Provide a SavePath object to get load paths")

            self.encoder.load_state_dict(torch.load(
                model_path + "/encoder_{}.pth".format(checkpoint)))
            self.decoder.load_state_dict(torch.load(
                model_path + "/decoder_{}.pth".format(checkpoint)))
            self.discriminator.load_state_dict(torch.load(
                model_path + "/discriminator_{}.pth".format(checkpoint)))
            self.reconstr_loss_epoch = np.loadtxt(
                list_path + '/reconstr_loss_{}.txt'.format(checkpoint)).tolist()

        if optimizer == "Adam":
            self.enc_optim = optim.Adam(self.encoder.parameters(),
                                        lr=self.args.lr)
            self.dec_optim = optim.Adam(self.decoder.parameters(),
                                        lr=self.args.lr)
            self.dis_optim = optim.Adam(self.discriminator.parameters(),
                                        lr=0.5 * self.args.lr)

            enc_scheduler = StepLR(self.enc_optim, step_size=30, gamma=0.5)
            dec_scheduler = StepLR(self.dec_optim, step_size=30, gamma=0.5)
            dis_scheduler = StepLR(self.dis_optim, step_size=30, gamma=0.5)

    def unfreeze_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def freeze_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def save_models(self, model_path, epoch_no):

        print("Saving models")
        torch.save(self.encoder.state_dict(),
                   model_path + "/encoder_{}.pth".format(epoch_no))
        torch.save(self.decoder.state_dict(),
                   model_path + "/decoder_{}.pth".format(epoch_no))
        torch.save(self.discriminator.state_dict(),
                   model_path + "/discriminator_{}.pth".format(epoch_no))

    def save_lists(self, list_path, epoch_no):
        print("Saving list")
        np.savetxt(list_path + '/reconstr_loss_{}.txt'.format(epoch_no),
                   self.reconstr_loss_epoch)

    def save_images(self, recimage, sampimage, image_path, epoch_no):
        print("Saving images")
        save_image(recimage, image_path +
                   '/inputs_reconstr_{}.png'.format(epoch_no))

        save_image(sampimage, image_path +
                   '/sample_{}.png'.format(epoch_no))

    def train(self, out_frequency, save_model_frequency, save_paths):

        chkpt = self.checkpoint if self.checkpoint else 0

        reconstr_loss = []
        image_path, list_path, model_path = save_paths.get_save_paths()

        one = torch.tensor(1)
        mone = one * -1

        for epoch in range(chkpt, chkpt + self.args.n_epochs):
            for step, (images, _) in tq(enumerate(self.train_loader)):

                reconstr_loss.clear()
                # discriminator_loss.clear()

                if torch.cuda.is_available():
                    images = images.cuda()

                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.discriminator.zero_grad()

                # ======== Train Discriminator ======== #

                self.freeze_params(self.decoder)
                self.freeze_params(self.encoder)
                self.unfreeze_params(self.discriminator)

                z_fake = torch.randn(images.size()[0], self.args.n_z) \
                         * self.args.sigma

                if torch.cuda.is_available():
                    z_fake = z_fake.cuda()

                d_fake = self.discriminator(z_fake)

                z_real = self.encoder(images)
                d_real = self.discriminator(z_real)

                torch.log(d_fake).mean().backward(mone)
                torch.log(1 - d_real).mean().backward(mone)

                self.dis_optim.step()

                # ======== Train Generator ======== #

                self.unfreeze_params(self.decoder)
                self.unfreeze_params(self.encoder)
                self.freeze_params(self.discriminator)

                batch_size = images.size()[0]

                z_real = self.encoder(images)
                x_recon = self.decoder(z_real)
                d_real = self.discriminator(self.encoder(Variable(images.data)))

                recon_loss = self.criterion(x_recon, images)
                d_loss = self.args.LAMBDA * (torch.log(d_real)).mean()

                recon_loss.backward(one)
                d_loss.backward(mone)

                self.enc_optim.step()
                self.dec_optim.step()

                reconstr_loss.append(recon_loss.data.item())

            if (epoch + 1) % out_frequency == 0:
                print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f"
                      % (epoch + 1, self.args.n_epochs, step + 1,
                         len(self.train_loader), recon_loss.data.item()))

            if (epoch + 1) % out_frequency == 0:
                self.reconstr_loss_epoch.append(np.mean(reconstr_loss))

                batch_size = self.args.batch_size
                test_iter = iter(self.test_loader)
                test_data = next(test_iter)

                z_real = self.encoder(Variable(test_data[0]).cuda())
                reconst = decoder(z_real).cpu().view(batch_size, 1, 28, 28)
                sampimage = self.decoder(torch.randn_like(z_real)).cpu().view(
                    batch_size, 1, 28, 28)
                recimage = torch.cat((test_data[0].view(batch_size, 1, 28, 28),
                                      reconst.data), axis=3)

                self.save_images(recimage, sampimage, image_path, epoch + 1)

                # save_image(test_data[0].view(batch_size, 1, 28, 28),
                #                 image_path + '/input_{}.png'.format(epoch+1))
                # save_image(reconst.data, image_path +
                #                         '/reconstr_{}.png'.format(epoch + 1))

            if (epoch + 1) % save_model_frequency == 0:
                self.save_models(model_path, epoch + 1)
                self.save_lists(list_path, epoch + 1)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

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
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
