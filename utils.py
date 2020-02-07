
import matplotlib.pyplot as plt
import torch
from termcolor import colored
import os.path
import imgaug.augmenters as iaa
import numpy as np


# learning-rate decay
def lr_decay(lr: float=0, epoch: int=0, decay_rate: float=0.1, period: int=5) -> float:
    if epoch % period == 0:
        return lr * (1 / (decay_rate * epoch + 1))
    else:
        return lr

# augmentate image with a 50% flip chance
def rand_flip(matrix):
    random_flip = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
    return random_flip(images=matrix)

# plot loss history of discriminator and generator
def plot_loss_history(disc_loss: float, disc_real_loss: float, disc_fake_loss: float, gen_loss: float, save_to: str):
    fig, axs = plt.subplots(2, 2)

    axs[0][0].plot(range(len(disc_loss)), disc_loss, c="r")
    axs[0][0].set_title("disc.-loss")

    axs[0][1].plot(range(len(gen_loss)), gen_loss, c="b")
    axs[0][1].set_title("gen.-loss")

    axs[1][0].plot(range(len(disc_loss)), disc_loss, c="r")
    axs[1][0].plot(range(len(gen_loss)), gen_loss, c="b")
    axs[1][0].set_title("disc. (r) and gen. (b) losses")

    axs[1][1].plot(range(len(disc_real_loss)), disc_real_loss, c="g")
    axs[1][1].plot(range(len(disc_fake_loss)), disc_fake_loss, c="y")
    axs[1][1].set_title("disc.-real (g) and disc.-fake (y) losses")
    plt.savefig(save_to)
    plt.show()

# print prettifed trainings progress
def print_progress(epoch: int, epochs: int, disc_loss: float, disc_real_loss: float, disc_fake_loss: float, gen_loss: float, disc_lr: float, gen_lr: float):
    epoch = colored((epoch + 1), "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    disc_loss = colored(disc_loss, "cyan", attrs=['bold'])
    disc_real_loss = colored(disc_real_loss, "cyan", attrs=['bold'])
    disc_fake_loss = colored(disc_fake_loss, "cyan", attrs=['bold'])
    gen_loss = colored(gen_loss, "cyan", attrs=['bold'])
    disc_lr = colored(disc_lr, "cyan", attrs=["bold"])
    gen_lr = colored(gen_lr, "cyan", attrs=["bold"])

    print(" ")
    print("\nepoch {} - disc_loss: {} - disc_real_loss: {} - disc_fake_loss: {} - gen_loss: {}".format(epoch, disc_loss, disc_real_loss, disc_fake_loss, gen_loss))
    print("disc_lr: {} - gen_lr: {}".format(disc_lr, gen_lr))
    print("\n........................................................................................................................\n")

# add gaussian noise
def add_gaussian_noise(x, noise_rate: float=0.0):
    x += noise_rate * torch.randn(1, 28, 28).cuda()
    return x

# plot generated images
def show_generated(model, view_seconds: int=3, current_epoch: int=1, period: int=5, save_to: str=""):
    if current_epoch % period == 0:
        noise_vector = torch.rand((100, 1, 1)).cuda().reshape(1, 100, 1, 1)
        fake_image = model.eval()(noise_vector)

        plt.matshow(fake_image.cpu().detach().numpy().reshape(28, 28))
        plt.savefig(save_to)

        plt.show(block=False)
        plt.pause(view_seconds)
        plt.close()

# save both models
def save_models(generator, discriminator, save_to: str="", current_epoch: int=1, period: int=5):
    if current_epoch % period == 0:
        torch.save(generator.state_dict(), (save_to + "/" + "generator.pt"))
        torch.save(discriminator.state_dict(), (save_to + "/" + "discriminator.pt"))


