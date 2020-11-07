import os
import time


class SavePath:

    def __init__(self, args, checkpoint_path=None):
        """
        Object to generate a save path
        :param args: TrainConfig object use for base_path
        :param checkpoint_path: str, use to load paths from an existing directory

        TODO:
            Probably change args to base_path
        """
        self.args = args
        if checkpoint_path:
            self.results_path = checkpoint_path
        else:
            timestamp = time.asctime(time.localtime(time.time())).replace(" ", "-").replace(":", "-")
            self.results_path = self.args.base_path + "outs/{}/".format(timestamp)

        print(self.results_path)

    def get_save_paths(self, make_directories=True):

        image_path = self.results_path + "images/"
        list_path = self.results_path + "lists/"
        model_path = self.results_path + "models/"

        if make_directories:
            os.makedirs(image_path, exist_ok=True)
            os.makedirs(list_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)

        return image_path, list_path, model_path
