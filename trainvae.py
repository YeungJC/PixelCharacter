import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Assumes dataset has been downloaded from kaggle
# these commands open up the zip file. the characters dataset is in the 'data' folderd

import zipfile

def extract_files():
    zip_ref = zipfile.ZipFile("pixel-characters-dataset.zip", "r")
    zip_ref.extractall("./content")
    zip_ref.close()


# import libraries

# math is a basic python library for mathematical operations
import math

# numpy is library for working with arrays
import numpy as np

# pandas is a library for working with tables
import pandas as pd

# matplotlib is used for plotting graphs as well as images
import matplotlib.pyplot as plt

# pytorch is our deep learning library
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.nn import init

# torchvision is part of pytorch but contains specific functions for handling image data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# set default configurations for training the neural network
# try out different values to get the best result. this default is definitely not the best!
BATCH_SIZE = 32  # The number of images shown to the model per iteration
LEARNING_RATE = 1e-3  # Amount weights are adjusted each iteration
NUM_EPOCHS = 40  # Number of complete training cycles
BETA = 200000  # Increasing beta increases weighting of recontruction loss

def load_transform_images():
# these are transformations we do to the images in our data folder, before we pass them to the neural network for training
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  #(64,64,3) -> (3, 64, 64)
            transforms.Resize((64, 64), antialias=True),  # (3, 32, 32)
            torchvision.transforms.RandomHorizontalFlip(
                p=0.5
            ),  #  50% chance to flip an image horizontally during training
        ]
    )

    # to train a neural network, we need to show the neural network images many times. the dataset and dataloader handles this
    train_set = ImageFolder(root="./content/data", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    return train_loader

# Dataset (train_set) -> gathers data -> processes one image
# Dataloader (train_loader) -> takes images, and put them into a batch to show them to the model

# variational autoencoder model architecture
# there are two parts to the architecture:
# 1. encoder
# 2. decoder
from models import VariationalEncoder, Decoder

def initialise_model(train_loader):

    # this creates our encoder
    encoder = VariationalEncoder().to(device)

    # pass dummy input into encoder so that we can get the final shape of the latent vector created from the encoder
    dummy_input, dummy_labels = next(iter(train_loader))
    _ = encoder(dummy_input.to(device))

    # this is the final shape of the latent vector
    shape_before_flattening = encoder.shape_before_flattening

    # we need that so we can tell the decoder what shape to use
    decoder = Decoder(shape_before_flattening=shape_before_flattening).to(device)
    return encoder, decoder



# define our accuracy metric
def accuracy(out, yb):
    return (out.argmax(dim=1) == yb).float().mean()

def train_model(encoder, decoder, train_loader):
    # define optimiser
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE
    )

    # training loop
    for epoch in range(NUM_EPOCHS):
        train_epoch_loss = 0.0
        encoder.train()
        decoder.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero optimizer
            optimizer.zero_grad()
            # forward pass
            encoded = encoder(inputs)
            outputs = decoder(encoded)
            # loss
            loss = (
                BETA * nn.MSELoss()(outputs, inputs) + encoder.kl_loss
            )  # here is where BETA is used. it changes the weighting of the loss
            # backward pass
            loss.backward()
            # optimize
            optimizer.step()

            train_epoch_loss += loss
        train_epoch_loss /= len(train_loader)

        print(f"Epoch {epoch+1} Train loss: {train_epoch_loss:.3f}")


    print("Finished Training")
    return decoder

def save_model(decoder):
    torch.save(decoder.state_dict(), "decoder.pt")

# generate random image
def show_images(decoder):
    # increase variation to increase how much variation in images created. Too big of a number will create bad images so there is a balance
    variation = 2

    # set model to inference mode
    decoder.eval()

    # also related to not training the model - no grad means no gradient
    with torch.no_grad():
        random_latent_vector = torch.randn(1, 40) * variation
        random_image = decoder(random_latent_vector.to(device))
        random_image = random_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    plt.imshow(random_image)
    plt.show()

    # interpolate between two images part 1

    # increase variation to increase how much variation in images created. 
    variation = 2

    # set model to inference mode
    decoder.eval()

    with torch.no_grad():
        # generate random first image
        random_latent_vector_1 = torch.randn(1, 40) * variation
        random_image_1 = decoder(random_latent_vector_1.to(device))
        random_image_1 = random_image_1.permute(0, 2, 3, 1).squeeze().cpu().numpy()

        # generate random second image
        random_latent_vector_2 = torch.randn(1, 40) * variation
        random_image_2 = decoder(random_latent_vector_2.to(device))
        random_image_2 = random_image_2.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    # plot both images
    # plot interpolations
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(random_image_1)
    ax[0].axis("off")

    ax[1].imshow(random_image_2)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    # interpolate between two images part 2

    # pick a value between 0.0 (image 1) and 1.0 (image 2) to interpolate
    interpolation_value = 0.5

    # remember, we are interpolating in the latent space, not mixing the actual images. 
    random_latent_vector_3 = ((1 - interpolation_value) * random_latent_vector_1) + (
        interpolation_value * random_latent_vector_2
    )

    with torch.no_grad():
        # generate random mixed image
        random_image_3 = decoder(random_latent_vector_3.to(device))
        random_image_3 = random_image_3.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    plt.imshow(random_image_3)
    plt.show()

def main():
    extract_files()
    train_loader = load_transform_images() 
    encoder, decoder = initialise_model(train_loader)
    decoder = train_model(encoder, decoder, train_loader)
    save_model(decoder)
    show_images(decoder)


# Run main function
main()