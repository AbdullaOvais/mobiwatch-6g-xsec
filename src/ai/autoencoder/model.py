import numpy as np
from torch import nn

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        latent_dim=None,
        architecture="kitsune"
    ):
        super(Autoencoder, self).__init__()

        # ---------------------------------
        # OPTION 1: Kitsune-style (default)
        # ---------------------------------
        if architecture == "kitsune":

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, int(input_dim * 0.75)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.75), int(input_dim * 0.5)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.5), int(input_dim * 0.25)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.25), int(input_dim * 0.1))
            )

            self.decoder = nn.Sequential(
                nn.Linear(int(input_dim * 0.1), int(input_dim * 0.25)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.25), int(input_dim * 0.5)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.5), int(input_dim * 0.75)),
                nn.ReLU(True),
                nn.Linear(int(input_dim * 0.75), input_dim),
            )

        # ---------------------------------
        # OPTION 2: Custom configurable AE
        # ---------------------------------
        elif architecture == "custom":

            if hidden_dim is None or latent_dim is None:
                raise ValueError("hidden_dim and latent_dim must be provided for custom architecture")

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(True),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, input_dim),
            )

        else:
            raise ValueError("Unknown architecture type")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x