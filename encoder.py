import torch
import torch.nn as nn
import torch.nn.functional as F

#Bidirectional Encoder as explained in section 3.1
# Two-layer bidirectional LSTM network by Schuster, M. and Paliwal, K. K. Bidirectional recurrent neural networks. 1997.
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=2048, latent_dim=512):

        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        # Bidirectional LSTM layers with two layers
        self.lstm = nn.LSTM(batch_first=True,input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        
        # Two-Fully connected layers to produce µ and σ
        self.fc_mu = nn.Linear(2 * hidden_size, latent_dim)
        self.fc_sigma = nn.Linear(2 * hidden_size, latent_dim)
       
        
    def forward(self, input_sequence):

        
        # LSTM forward pass
        output, _ = self.lstm(input_sequence)
   
        
        # No need to concatenate, the final output contain a concatenation of the forward and reverse hidden states at each time step in the sequence, thus we need to get the last one
        h_T = output[:,-1,:]
        
        
        mu = self.fc_mu(h_T) # W_hµ * h_T + b_µ (Equation (6))
        sigma = torch.log(torch.exp(self.fc_sigma(h_T)) + 1) # log (exp(W_hσ * h_T + b_σ) + 1) (Equation (7))


        # Transform to MultivariateNormalDiag distribution
        # where eps ~ N(0, 1)

        # Prevent gradient flowing through sampling during training because of randomness
        with torch.no_grad():
           
            eps = torch.randn_like(sigma)

        
        z = mu + eps*sigma  # ε ∼ N (0, I), z = µ + σ ⊙ ε (Equation (2))

        # mu: torch.Tensor[batch_size,latent_dim]
        # sigma: torch.Tensor[batch_size,latent_dim]
        # z: torch.Tensor[batch_size,latent_dim]
        return mu, sigma, z


# if __name__ == '__main__':
#     e = Encoder(27)
#     seq_length = 64
#     temp = torch.rand(512,seq_length,27)

#     mu,sigma,z = e(temp)

#     print(mu.shape, sigma.shape,z.shape)