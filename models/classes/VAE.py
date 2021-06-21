import ipdb
import torch
import os
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, model_path, num_ind, num_exp_level, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_size
        self.desc = desc
        self.model_path = model_path
        self.max_len = hp.max_len
        self.num_ind = num_ind
        self.num_exp_level = num_exp_level

        self.encoder = None
        self.decoder = None

    def forward(self):
        ipdb.set_trace()

    def inference(self,):
        ipdb.set_trace()

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)


    def training_step(self):
        ipdb.set_trace()


    def validation_step(self):
        ipdb.set_trace()


    def test_epoch_start(self):
        ipdb.set_trace()

    def test_step(self):
        ipdb.set_trace()

    def test_epoch_end(self):
        ipdb.set_trace()




