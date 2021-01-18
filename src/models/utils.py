from pytorch_model_summary import summary
import torch


def save_model_architectures(results_path, encoders, decoders, discriminator, encoder_input_shapes: list,
                             latent_input_shape):

    """
    :param results_path: str
    :param encoders: list of encoders of type nn.Module
    :param decoders: list of nn.Module
    :param discriminator: nn.Module
    :param encoder_input_shapes: list of tuples
    :param latent_input_shape: tuple
    :return:
    """

    file_name = results_path + "model_summaries.txt"
    with open(file_name, "w") as text_file:
        for i, encoder in enumerate(encoders):
            print("Encoder_{}".format(i), file=text_file)
            input_tensor = torch.zeros((encoder_input_shapes[i]))
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            print(summary(encoder, input_tensor, show_input=True, show_hierarchical=True),
                  file=text_file)

            latent_input = torch.zeros(latent_input_shape)
            if torch.cuda.is_available():
                latent_input = latent_input.cuda()
            for i, decoder in enumerate(decoders):
                print("Decoder{}".format(i), file=text_file)
                print(summary(decoder, latent_input, show_input=True, show_hierarchical=True),
                      file=text_file)

            print("Discriminator{}".format(i), file=text_file)
            print(summary(discriminator, latent_input, show_input=True, show_hierarchical=True),
                  file=text_file)


def save_models(model_path, epoch_no, encoders, decoders, discriminator):

    """
    :param model_path: str
    :param epoch_no: int
    :param encoders: list of nn.Module
    :param decoders: list of nn.Module
    :param discriminator: nn.Module
    :return:
    """

    for i, encoder in enumerate(encoders):
        torch.save(encoder.state_dict(), model_path + "/encoder_{}_epoch_{}.pth".format(i + 1, epoch_no))
    for i, decoder in enumerate(decoders):
        torch.save(decoder.state_dict(), model_path + "/decoder_{}_epoch_{}.pth".format(i + 1, epoch_no))
    torch.save(discriminator.state_dict(), model_path + "/discriminator_epoch_{}.pth".format(epoch_no))


def load_model_weights(model_path, encoders, decoders, discriminator, checkpoint):

    """
    :param model_path: str
    :param encoders: list(nn.Module)
    :param decoders: list(nn.Module)
    :param discriminator: nn.Module
    :param checkpoint: int
    :return:
    """

    for i, encoder in enumerate(encoders):
        encoders[i].load_state_dict(torch.load(model_path + "/encoder_{}_epoch_{}.pth".format(i + 1, checkpoint)))
    for i, decoder in enumerate(decoders):
        decoders[i].load_state_dict(torch.load(model_path + "/decoder_{}_epoch_{}.pth".format(i + 1, checkpoint)))
    discriminator.load_state_dict(torch.load(model_path + "/discriminator_epoch_{}.pth".format(checkpoint)))
