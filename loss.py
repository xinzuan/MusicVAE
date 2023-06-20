from torch.nn.functional import binary_cross_entropy
from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch

def ELBO_loss(y, t, mu, sigma, beta):
    # log[p(x|z)]
    likelihood = -binary_cross_entropy(y, t, reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Kullback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    n_mu = torch.Tensor([0])
    n_sigma = torch.Tensor([1])

    p = Normal(n_mu, n_sigma)
    q = Normal(mu, sigma)

    #KL divergence
    kl_div = kl_divergence(q, p)

    # common modification to the ELBO introduces the KL weight hyperparameter β (Bowman et al., 2016; Higgins et al., 2017) 
    elbo = torch.mean(likelihood) - (beta * torch.mean(kl_div))  # Equation (3)


    return -elbo, kl_div.mean(), beta * kl_div.mean()  