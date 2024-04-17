from itertools import pairwise

import torch
import torch.nn as nn

class encoder_block(nn.Module):
    '''
    Encoder for 64x64 face generation. The hidden dimensions can be tuned.
    '''
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.relu(x)
        return x

class decoder_block(nn.Module):
    '''
    Decoder for 64x64 face generation. The hidden dimensions can be tuned.
    '''
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.Tconv = nn.ConvTranspose2d(in_channel, out_channel, 
                                       kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.Tconv(x)
        x = self.BN(x)
        x = self.relu(x)
        return x
    
    
class picVAE(nn.Module):
    '''
    VAE for 64x64 face generation. The hidden dimensions can be tuned.
    '''
    def __init__(self, hiddens=[16, 32, 64, 128, 256], img_length = 64, latent_dim=128) -> None:
        super().__init__()

        # encoder
        modules = [encoder_block(3, hiddens[0])] # 3 channels for RGB to start
        cur_length = img_length // 2
        for in_channel, out_channel in pairwise(hiddens):
            modules.append(encoder_block(in_channel, out_channel))
            cur_length = cur_length // 2

        self.encoder = nn.Sequential(*modules)

        # calculate mean and variance
        self.mean_linear = nn.Linear(hiddens[-1] * cur_length * cur_length,
                                     latent_dim)
        self.var_linear = nn.Linear(hiddens[-1] * cur_length * cur_length,
                                    latent_dim)
        self.latent_dim = latent_dim

        # change vector to image
        self.decoder_projection = nn.Linear(latent_dim, 
                                            hiddens[-1] * cur_length * cur_length)
        self.decoder_input_chw = (hiddens[-1], cur_length, cur_length)
        # decoder
        modules = []
        for in_channel, out_channel in pairwise(hiddens[::-1]):
            modules.append( decoder_block(in_channel, out_channel) )
            cur_length = cur_length * 2
        self.decoder = nn.Sequential(*modules)

        # change channel to 3
        assert cur_length == img_length // 2
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hiddens[0],
                                hiddens[0],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
            nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
            nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        mean = self.mean_linear(x)
        eps = torch.randn_like(mean)
        logvar = self.var_linear(x)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        x = self.decoder(x)
        x = self.final_layer(x)
        return x, mean, logvar
    
    def sample(self, device='cpu'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    
if __name__ == '__main__':
    model = picVAE()
    x = torch.randn(16, 3, 64, 64)
    y, mean, logvar = model(x)
    print('x:', x.shape)
    print('y:', y.shape)
    print(mean.shape)
    print(logvar.shape)
    print(model.sample().shape)