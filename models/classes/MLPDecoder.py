import torch
import pytorch_lightning as pl


class MLPDecoder(pl.LightningModule):
    def __init__(self, latent_size, hidden_size, output_size, hp):
        super().__init__()
        self.hp = hp
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        self.in_layer = torch.nn.Linear(latent_size, hidden_size)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        outs = self.leaky_relu(self.in_layer(inputs))
        outs_2 = self.leaky_relu(self.hidden_layer(outs))
        return torch.sigmoid(self.out_layer(outs_2))

