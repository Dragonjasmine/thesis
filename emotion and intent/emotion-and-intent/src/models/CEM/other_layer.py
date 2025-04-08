import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import config


class Sampling(nn.Module):
    def __init__(self,hidden_dim, posterior_hidden_dim, out_dim=32):
        super(Sampling, self).__init__()

        #prior
        self.h_prior = nn.Linear(hidden_dim, hidden_dim)
        self.mu_prior = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_prior = nn.Linear(hidden_dim, hidden_dim)
        self.dense_z_prior = nn.Linear(hidden_dim, out_dim)

        #posterior
        self.h_posterior = nn.Linear(hidden_dim, hidden_dim)
        self.mu_posterior = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_posterior = nn.Linear(hidden_dim, hidden_dim)
        self.dense_z_posterior = nn.Linear(hidden_dim, out_dim)

    def prior(self, x):
        h1 = F.relu(self.h_prior(x))
        mu_state = self.mu_prior(h1)
        logvar_state = self.logvar_prior(h1)
        return mu_state, logvar_state

    def posterior(self, response):
        #x = torch.cat([context, response], dim=-1)
        x = response
        h_posterior = F.relu(self.h_posterior(x))
        mu_posterior = self.mu_posterior(h_posterior)
        logvar_posterior = self.logvar_posterior(h_posterior)
        return mu_posterior, logvar_posterior

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, q_h):
        x = q_h
        mu_prior, logvar_prior = self.prior(x)
        z_prior = self.reparameterize(mu_prior, logvar_prior)
        E_prob_p = torch.softmax(self.dense_z_prior(z_prior), dim=-1)  # (bs, len(pos))
        return mu_prior, logvar_prior, z_prior, 0

    def forward_train(self, response):
        mu_posterior, logvar_posterior = self.posterior(response)
        z_posterior = self.reparameterize(mu_posterior, logvar_posterior)
        E_prob_posterior = torch.softmax(self.dense_z_posterior(z_posterior), dim=-1)
        return mu_posterior, logvar_posterior, z_posterior, 0

    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0]).to(config.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(config.device)
            logvar_prior = torch.FloatTensor([0.0]).to(config.device)
        kl_div = torch.sum(
            0.5
            * (
                logvar_prior
                - logvar_posterior
                + (logvar_posterior.exp() + (mu_posterior - mu_prior).pow(2))
                / logvar_prior.exp()
                - one
            )
        )
        return kl_div

