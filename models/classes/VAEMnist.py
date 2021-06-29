import ipdb
import torch
import pytorch_lightning as pl
from data.visualisation import tsne_in_vae_space
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from utils.models import masked_softmax, plot_grad_flow
import models.classes


class VAEMnist(pl.LightningModule):
    def __init__(self, datadir, emb_dim, desc, model_path, num_classes, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_dim
        self.desc = desc
        self.model_path = model_path
        self.max_len = hp.max_len
        self.num_classes = num_classes
        self.max_len = hp.max_len
        self.log_scale = torch.nn.Parameter(torch.Tensor([self.hp.logscale]))

        input_size = emb_dim + num_classes

        self.vae_encoder = models.classes.MLPEncoder(input_size, hp.mlp_hs, hp.latent_size, hp)
        self.vae_decoder = models.classes.MLPDecoder(hp.latent_size, hp.mlp_hs, emb_dim, hp)

    def forward(self, images, labels):
        inputs = self.get_vae_encoder_input(images, labels)
        mu_enc, log_var_enc = self.vae_encoder(inputs)
        std = torch.exp(log_var_enc / 2)
        z_dist = Normal(mu_enc, std + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae_decoder(dec_input)

        loss_vae_rec, loss_vae_kl = self.loss_function(reconstructed_input,
                                                       images, mu_enc, log_var_enc)
        ref_dist = Normal(torch.zeros(mu_enc.shape[0], mu_enc.shape[-1]).cuda(),
                          torch.ones(mu_enc.shape[0], mu_enc.shape[-1]).cuda())
        other_kl = torch.sum(kl_divergence(z_dist, ref_dist))
        ipdb.set_trace()
        # loss_vae_rec = torch.nn.functional.mse_loss(sent_embed[:, -1, :], reconstruc  ted_input, reduction="sum")
        # obs_distrib = Normal(reconstructed_input, torch.exp(self.log_scale))
        # loss_vae_rec = - obs_distrib.log_prob(sent_embed[:, -1, :]).sum()

        # ref_dist = Normal(torch.zeros(mu_enc.shape[0], mu_enc.shape[-1]).cuda(),
        #                   torch.ones(mu_enc.shape[0], mu_enc.shape[-1]).cuda())
        # loss_vae_kl =
        if torch.isnan(loss_vae_kl) or torch.isinf(loss_vae_kl):
            ipdb.set_trace()
        if torch.isnan(loss_vae_rec) or torch.isinf(loss_vae_rec):
            ipdb.set_trace()
        return loss_vae_rec, loss_vae_kl

    def inference(self, images, labels):
        inputs = self.get_vae_encoder_input(images, labels)
        mu_enc, log_var_enc = self.vae_encoder(inputs)
        std = torch.exp(log_var_enc / 2)
        z_dist = Normal(mu_enc, std + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae_decoder(dec_input)
        torch.ones(mu_enc.shape[0], mu_enc.shape[-1]).cuda()
        return reconstructed_input, images, z_dist, ref_dist

    def get_projection(self, images, labels):
        inputs = self.get_vae_encoder_input(images, labels)
        mu_enc, log_var_enc = self.vae_encoder(inputs)
        std = torch.exp(log_var_enc / 2)
        z_dist = Normal(mu_enc, std + 1e-10)
        dec_input = z_dist.rsample()
        return dec_input

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        images, labels = batch[0], batch[1]
        sample_len = len(images)
        rec, kl = self.forward(images, labels)
        train_kl_loss = self.hp.coef_kl * kl / sample_len
        train_rec_loss = self.hp.coef_rec * rec / sample_len
        train_loss = train_rec_loss + train_kl_loss
        self.log('train_rec_loss', train_rec_loss, on_step=True, on_epoch=False)
        self.log('train_kl_loss', train_kl_loss, on_step=True, on_epoch=False)
        self.log('train_loss', train_loss, on_step=True, on_epoch=False)
        if self.hp.plot_grad == "True":
            train_loss.backward()
            plot_grad_flow(self.vae_encoder.named_parameters(), self.desc + "enc")
            plot_grad_flow(self.vae_decoder.named_parameters(), self.desc + "dec")
            ipdb.set_trace()

        return {"loss": train_loss}

    def validation_step(self, batch, batch_nb):
        images, labels = batch[0], batch[1]
        sample_len = len(images)
        rec, kl = self.forward(images, labels)
        val_kl_loss = self.hp.coef_kl * kl / sample_len
        val_rec_loss = self.hp.coef_rec * rec / sample_len
        val_loss = val_rec_loss + val_kl_loss
        self.log('val_rec_loss', val_rec_loss, on_step=False, on_epoch=True)
        self.log('val_kl_loss', val_kl_loss, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_step_outputs):
        if self.hp.plot_latent_space == "True":
            tsne_in_vae_space.main(self.hp, self, self.desc, self.trainer.current_epoch, "mnist")

    def test_epoch_start(self):
        ipdb.set_trace()

    def test_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        reconstructed_input, tgt_input, z_dist, ref_dist = self.inference(sentences, ind_indices, exp_indices)
        loss_vae_rec = torch.nn.functional.mse_loss(tgt_input, reconstructed_input)
        loss_vae_kl = torch.mean(kl_divergence(z_dist, ref_dist))
        ipdb.set_trace()
        return {"kl_div": loss_vae_kl, "mse_loss": loss_vae_rec}

    def test_epoch_end(self):
        ipdb.set_trace()

    def get_vae_encoder_input(self, img, labels):
        reshaped_img = img.view(self.hp.b_size, -1)
        inpt = torch.cat([reshaped_img, labels], dim=-1)
        return inpt

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE, KLD