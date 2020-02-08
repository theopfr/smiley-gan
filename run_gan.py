
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import lr_scheduler

from generator import Generator
from discriminator import Discriminator
from utils import lr_decay, plot_loss_history, show_generated, save_models, print_progress, rand_flip
from benchmarks import Benchmark

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os


class RunGAN:
    def __init__(self, doodle_dataset, epochs: int=50, batch_size: int=16, disc_lr: float=0.001, gen_lr: float=0.001, 
                                    disc_lr_decay: float=0.1, gen_lr_decay: float=0.1, lr_decay_period=10, gaussian_noise_range: tuple=(0.5, 0.1)):
        # dataset
        self.train_set, self.test_set, self.val_set = doodle_dataset[0], doodle_dataset[1], doodle_dataset[2]

        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.disc_lr = disc_lr
        self.gen_lr = gen_lr
        self.disc_lr_decay = disc_lr_decay
        self.gen_lr_decay = gen_lr_decay
        self.lr_decay_period = lr_decay_period
        self.gaussian_noise_range = gaussian_noise_range

        # discriminator model
        self.discriminator = Discriminator().cuda()

        # generator model
        self.generator = Generator().cuda()

        # number of batches per epoch (= iterations per epoch)
        self.iterations = int(len(self.train_set) / self.batch_size) - 1
        
        # initialize benchmarker (clear_all=True deletes all previous benchmarks)
        self.benchmark_logger = Benchmark("benchmarks/benchmarks.json", clear_all=False)
        self.benchmark_id = self.benchmark_logger.create_id()

        # benchmark files
        self.generated_images_path = "benchmarks/generated_images/" + self.benchmark_id
        self.models_path = "benchmarks/models/" + self.benchmark_id
        self.plots_path = "benchmarks/plots"

        # create benchmark folders
        os.mkdir(self.generated_images_path)
        os.mkdir(self.models_path)


    """ create a batch fake images G(z) with the generator G, where z are noise vectors and add labels [0, 1] """
    def _create_fake_image_batch(self) -> list:
        noise_inputs = torch.cat([torch.rand((100, 1, 1)).cuda() for _ in range(self.batch_size)], dim=0).reshape(self.batch_size, 100, 1, 1)

        fake_images = self.generator.train()(noise_inputs)
        
        fake_image_batch = []
        for image in fake_images:
            fake_image_batch.append((image.cuda(), torch.Tensor([0]).cuda()))

        return fake_image_batch

    """ create a batch of true images from the dataset and add label [1, 0] """
    def _create_true_image_batch(self, batch_index: int) -> list:
        # random flip, soft labels
        true_image_batch = [(torch.Tensor(rand_flip(sample).reshape(1, 28, 28)).cuda(), torch.Tensor([np.random.uniform(0.85, 1.0)]).cuda()) \
                            for sample in self.train_set[(self.batch_size * batch_index):(self.batch_size * (batch_index + 1))]] 

        return true_image_batch

    """ create two batches """
    def _create_batch(self, iteration: int): 
        true_images = self._create_true_image_batch(iteration)
        fake_images = self._create_fake_image_batch() 

        images_real, targets_real, images_fake, targets_fake = [], [], [], []
        for idx in range(len(true_images)):
            images_real.append(true_images[idx][0])
            targets_real.append(true_images[idx][1])

            images_fake.append(fake_images[idx][0])
            targets_fake.append(fake_images[idx][1])

        images_real = torch.cat(images_real, dim=0).reshape((self.batch_size), 1, 28, 28)
        targets_real = torch.cat(targets_real, dim=0).reshape((self.batch_size), 1)

        images_fake = torch.cat(images_fake, dim=0).reshape((self.batch_size), 1, 28, 28)
        targets_fake = torch.cat(targets_fake, dim=0).reshape((self.batch_size), 1)

        return images_real, targets_real, images_fake, targets_fake

    """ train model """
    def train(self, plot_period: int=5):
        """ define loss-, optimzer- and scheduler-functions """
        criterion_disc = nn.BCELoss()
        criterion_gen = nn.BCELoss()

        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(0.5, 0.999))
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(0.5, 0.999))

        """ create benchmark """
        self.benchmark_logger.create_entry(self.benchmark_id, optimizer_disc, criterion_disc, self.epochs, self.batch_size, self.disc_lr, self.gen_lr, self.disc_lr_decay, self.gen_lr_decay, self.lr_decay_period, self.gaussian_noise_range)

        # initial noise rate
        noise_rate = self.gaussian_noise_range[0]

        # total loss log
        loss_disc_history, loss_disc_real_history, loss_disc_fake_history = [], [], []
        loss_gen_history = []

        for epoch in range(self.epochs):
            # epoch loss log
            epoch_loss_disc, epoch_loss_disc_real, epoch_loss_disc_fake = [], [], []
            epoch_loss_gen = []

            for iteration in tqdm(range(self.iterations), ncols=120, desc="batch-iterations"):
                images_real, targets_real, images_fake, targets_fake = self._create_batch(iteration)

                """ train discriminator """
                # update every third iteration to make the generator stronger
                self.discriminator.zero_grad()

                # train with real images
                predictions_real = self.discriminator.train()(images_real, gaussian_noise_rate=noise_rate)
                loss_real = criterion_disc(predictions_real, targets_real)

                loss_real.backward()

                # train with fake images
                predictions_fake = self.discriminator.train()(images_fake, gaussian_noise_rate=noise_rate)
                loss_fake = criterion_disc(predictions_fake, targets_fake)

                loss_fake.backward(retain_graph=True)

                if iteration % 1 == 0:
                    optimizer_disc.step()

                # save losses
                epoch_loss_disc.append(loss_real.item() + loss_fake.item())
                epoch_loss_disc_real.append(loss_real.item())
                epoch_loss_disc_fake.append(loss_fake.item())

                """ train generator """
                self.generator.zero_grad()

                # train discriminator on fake images with target "real image" ([1, 0])
                predictions_fake = self.discriminator.train()(images_fake)
                loss_gen = criterion_gen(predictions_fake, targets_real)

                loss_gen.backward()
                optimizer_gen.step()

                epoch_loss_gen.append(loss_gen.item())
    

            """ linear gaussian noise decay for disc. inputs """
            noise_rate = np.linspace(self.gaussian_noise_range[0], self.gaussian_noise_range[1], self.epochs)[epoch]


            """ save models """
            save_models(self.generator, self.discriminator, save_to=(self.models_path), current_epoch=epoch, period=5)
            

            """ calculate average losses of the epoch """
            current_loss_disc, current_loss_disc_real, current_loss_disc_fake = round(np.mean(epoch_loss_disc), 4), round(np.mean(epoch_loss_disc_real), 4), round(np.mean(epoch_loss_disc_fake), 4)
            current_loss_gen = round(np.mean(epoch_loss_gen), 4)


            """ get learning-rate """
            current_disc_lr = round(optimizer_disc.param_groups[0]["lr"], 7)
            current_gen_lr = round(optimizer_gen.param_groups[0]["lr"], 7)


            """ learning-rate decay (set 'p' to 'False' for not doing lr-decay) """
            do = False
            if do:
                optimizer_disc.param_groups[0]["lr"] = lr_decay(lr=optimizer_disc.param_groups[0]["lr"], epoch=epoch, decay_rate=self.disc_lr_decay, period=self.lr_decay_period)
                optimizer_gen.param_groups[0]["lr"] = lr_decay(lr=optimizer_gen.param_groups[0]["lr"], epoch=epoch, decay_rate=self.gen_lr_decay, period=self.lr_decay_period)


            """ save losses for plotting """
            loss_disc_history.append(current_loss_disc); loss_disc_real_history.append(current_loss_disc_real); loss_disc_fake_history.append(current_loss_disc_fake)
            loss_gen_history.append(current_loss_gen)


            """ print trainings progress """
            print_progress(epoch, self.epochs, current_loss_disc, current_loss_disc_real, current_loss_disc_fake, current_loss_gen, current_disc_lr, current_gen_lr)


            """ plot generated images """
            if plot_period is not None:
                show_generated(self.generator, view_seconds=1, current_epoch=epoch, period=plot_period, save_to=(self.generated_images_path + "/" + str(epoch + 1) + ".png"))


        """ plot loss history """
        plot_loss_history(loss_disc_history, loss_disc_real_history, loss_disc_fake_history, loss_gen_history, save_to=(self.plots_path + "/" + self.benchmark_id + ".png"))


    """ test model """
    def test(self):
        pass



if __name__ == "__main__":
    doodle_dataset_path = "dataset/dataset.npy"
    doodle_dataset = np.load(doodle_dataset_path, allow_pickle=True)

    runGAN = RunGAN(doodle_dataset,
                    epochs=35,
                    batch_size=256,
                    disc_lr=0.0002,
                    gen_lr=0.0002,
                    disc_lr_decay=0.2,
                    gen_lr_decay=0.2,
                    lr_decay_period=6,
                    gaussian_noise_range=(0.5, 0.05))

    runGAN.train(plot_period=2)
    runGAN.test()











# lr_decay(optimizer_disc.param_groups[0]["lr"], optimizer_disc, epoch=epoch)
# lr_decay(optimizer_gen.param_groups[0]["lr"], optimizer_gen, epoch=epoch)



"""
# print example ('got' should be the opposite of 'is')
if iteration == self.iterations - 1:
    print("\n\nfake-example:")
    print("is ", np.mean(targets_fake.cpu().detach().numpy(), axis=0), "got", np.mean(predictions_fake.cpu().detach().numpy(), axis=0))
    print("real-example:")
    print("is ", np.mean(targets_real.cpu().detach().numpy(), axis=0), "got", np.mean(predictions_real.cpu().detach().numpy(), axis=0))

"""

# print("\ndisc. learning rate:", optimizer_disc.param_groups[0]["lr"], "| gen. learning rate:", optimizer_gen.param_groups[0]["lr"])
