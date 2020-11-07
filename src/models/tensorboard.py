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

    def save_values(self, epoch_no, values: dict):
        """
        All params except for epoch_no might be changed depending on the training
        :param epoch_no:
        :param values: dict of str -> values or dicts
        :return:
        """

        for tag, value in values.items():
            if type(value) == 'dict':
                self.writer.add_scalars(tag, value, epoch_no)
            else:
                self.writer.add_scalar(tag, value, epoch_no)
