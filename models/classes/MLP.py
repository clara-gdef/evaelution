import torch
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, input_size, output_size, num_layers, hp):
        super().__init__()
        self.hp = hp
        self.input_size = input_size
        self.output_size = output_size
        self.num_layer = num_layers

        layer_list = []
        for i in range(num_layers):
            layer_list.append(torch.nn.Linear(input_size, input_size))
            layer_list.append(torch.nn.ReLU(inplace=True))
        layer_list.append(torch.nn.Linear(input_size, output_size))
        self.layers = torch.nn.Sequential(*layer_list)

    def forward(self, inputs):
        return self.layers(inputs)
