import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, image_size=64, channels=3, embedding_dim=40):
        """ This is a convolutional neural network-based encoder for a variational autoencoder

        args:
            image_size: The size of the input image
            channels: The number of channels in the input image. 1 is for greyscale and 3 is for RGB images
            embedding_dim: The dimensions of the latent space
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1) #(64,64,3) -> (32,32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) #(32,32,32) -> (16,16,64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (8,8,128) 

        self.shape_before_flattening = None

        flattened_size = (image_size // 8) * (image_size // 8) * 128

        self.fc1 = nn.Linear(flattened_size, embedding_dim) # (8192,) -> (40,)
        self.fc2 = nn.Linear(flattened_size, embedding_dim) # (8192,) -> (40,)

    def forward(self, x):
        # Definition of the Variational Encoder
        # 3 layers of convolution layers followed by a relu
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        self.shape_before_flattening = x.shape[1:]

        x = nn.Flatten()(x)
        mu = self.fc1(x) # mu is the mean
        log_var = self.fc2(x) # log_var is the log of the variance

        # reparameterise
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # eps is the noise to be added
        sample = eps * std + mu

        self.kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence

        return sample
    
class Decoder(nn.Module):
    def __init__(self, shape_before_flattening, embedding_dim=40, channels=3):
        """ This is a convolutional neural network-based decoder for a variational autoencoder

        args:
            shape before flatenning: shape of transformed input image before being passed to linear layers
            embedding_dim: The dimensions of the latent space
            channels: The number of channels in the input image. 1 is for greyscale and 3 is for RGB images
            
        """
        super().__init__()

        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        self.reshape_dim = shape_before_flattening

        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1 # (8,8,128) -> (16,16,128)
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1 # (16,16,128) -> (32,32,64)
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1 # (32,32,64) -> (64,64,32)
        )

        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1) # (64,64,32) -> (64,64,3) 

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.reshape_dim)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        x = torch.sigmoid(self.conv1(x)) # sigmoid activation function transforms all the values to the range [0,1]
        return x