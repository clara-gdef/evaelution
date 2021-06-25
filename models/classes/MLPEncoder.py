import torch
import pytorch_lightning as pl


class MLPEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, latent_size, hp):
        super().__init__()
        self.hp = hp
        self.input_size = input_size
        self.latent_size = latent_size
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        self.in_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.mean_linear = torch.nn.Linear(hidden_size, latent_size)
        self.var_linear = torch.nn.Linear(hidden_size, latent_size)

    def forward(self, inputs):
        outs = self.leaky_relu(self.in_layer(inputs))
        outs_2 = self.leaky_relu(self.hidden_layer(outs))
        return self.mean_linear(outs_2), self.var_linear(outs_2)
