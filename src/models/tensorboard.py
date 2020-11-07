from torch.utils.tensorboard import SummaryWriter
import os
from shutil import rmtree


class TensorboardLogger:

    def __init__(self, logs_path):
        """
        :param logs_path: str
        creates and maintains a tensorboard writer which can be used to log various metrics and losses
        throughout training and testing
        to view the tensorboard while training:
        1. On local machine: run tensorboard --logdir <path to logs>
        2. On colab:
            a. %load_ext tensorboard
            b. %tensorboard --logdir <path to logs>

        TODO:
            Test rmtree functionality

        """
        self.logs_path = logs_path
        self.writer = self.get_summary_writer()

    def get_summary_writer(self):
        if os.path.exists(self.logs_path):
            rmtree(self.logs_path)
        return SummaryWriter(log_dir=self.logs_path)

    def save_values_to_tensorboard(self, epoch_no, reconstr_loss_mean, reg_dist_loss_mean, disc_loss_mean,
                                   norms_mean):
        """
        All params except for epoch_no might be changed depending on the training
        :param epoch_no:
        :param reconstr_loss_mean:
        :param reg_dist_loss_mean:
        :param disc_loss_mean:
        :param norms_mean:
        :return:
        """
        self.writer.add_scalars("Reconstruction_loss", {"conv": reconstr_loss_mean[0],
                                                        "linear": reconstr_loss_mean[1]}, epoch_no)
        self.writer.add_scalar("Regularization_loss/reg_loss", reg_dist_loss_mean[0], epoch_no)
        self.writer.add_scalar("Distance_loss/dist_loss", reg_dist_loss_mean[1], epoch_no)
        self.writer.add_scalars("Discriminator_loss", {"d_prior_distr": disc_loss_mean[0],
                                                       "d_linear": disc_loss_mean[1],
                                                       "d_conv": disc_loss_mean[2]}, epoch_no)
        self.writer.add_scalars("norms", {"prior_distr": norms_mean[0],
                                          "linear": norms_mean[1],
                                          "conv": norms_mean[2]}, epoch_no)
