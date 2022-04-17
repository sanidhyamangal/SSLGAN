"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import os

import matplotlib.pyplot as plt  # for plotting
import torch  # for pytorch based stuff
import torch.nn as nn  # for nn stuff
import torchvision.utils as vutils  # for plotting and other part
from torchvision import transforms  # for transforming the vision related ops


def create_dirs_if_not_exists(path) -> None:
    """function to create subdirs for loss plots"""
    _path = os.path.split(path)

    if _path[0] == "":
        return

    os.makedirs(_path[0], exist_ok=True)


def plot_gan_loss_plots(disc_loss: List[float], gen_loss: List[float],
                        plot_name: str) -> None:
    """function to plot the gan loss"""
    plt.plot(disc_loss, label="disc_loss")
    plt.plot(gen_loss, label="gen_loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    create_dirs_if_not_exists(plot_name)
    plt.savefig(plot_name)
    plt.clf()


def create_rot_transforms():
    return transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=(64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def generate_and_save_images(model: nn.Module,
                             test_input: torch.FloatTensor,
                             image_name: str = "generated.png",
                             multi_channel: bool = False,
                             show_image: bool = True) -> None:
    """
    A helper function for generating and saving images during training ops

    :param model: model which needs to be used for generation of images
    :param image_name: name of an image to save as png
    :param test_input: seed value which needs to be used for image generation
    :param multi_channel: multi_channel value for generation and saving of images
    :return: None
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    model.eval()
    with torch.no_grad():
        predictions = model(test_input)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if not multi_channel:
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        else:
            plt.imshow(((predictions[i] * 127.5) + 127.5) / 255.0)
        plt.axis('off')

    create_dirs_if_not_exists(image_name)
    plt.savefig(image_name)

    # show image only if flagged to true
    if show_image:
        plt.show()
