import torch
import torch.nn as nn

from encoder import Encoder
from decoder import HierarchicalDecoder

class VAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(VAE, self).__init__()
    
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self. encoder = Encoder(input_size=encoder_config['input_size'])
        self.decoder = HierarchicalDecoder(latent_dim=decoder_config['latent_dim'], output_size=decoder_config['output_size'])
        
    def forward(self,input_sequence):
        mu, sigma, z = self.encoder(input_sequence)
        output = self.decoder(z, input_sequence)
        return mu, sigma, z, output
