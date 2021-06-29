import ipdb
import torch
import os
import pytorch_lightning as pl
import pickle as pkl
from data.visualisation import tsne_in_vae_space
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from transformers import CamembertTokenizer, CamembertModel
from utils.models import masked_softmax, plot_grad_flow
import models.classes


class VAEemb(pl.LightningModule):
    def __init__(self, datadir, emb_dim, desc, model_path, num_ind, num_exp_level, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_dim
        self.desc = desc
        self.model_path = model_path
        self.max_len = hp.max_len
        self.num_ind = num_ind
        self.num_exp_level = num_exp_level
        self.max_len = hp.max_len
        self.att_type = hp.att_type
        self.log_scale = torch.nn.Parameter(torch.Tensor([self.hp.logscale]))

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.voc_size = self.tokenizer.vocab_size

        self.text_encoder = CamembertModel.from_pretrained('camembert-base')

        tgt_ind_file = f"ind_20_embeddings_fr.pkl"
        with open(os.path.join(self.datadir, tgt_ind_file), 'rb') as f_name:
            ind_weights = pkl.load(f_name)
        tgt_exp_file = f"exp_3_embeddings_fr.pkl"
        with open(os.path.join(self.datadir, tgt_exp_file), 'rb') as f_name:
            exp_weights = pkl.load(f_name)

        self.ind_emb = torch.nn.Embedding.from_pretrained(ind_weights)
        self.exp_emb = torch.nn.Embedding.from_pretrained(exp_weights)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.ind_emb.parameters():
            param.requires_grad = False
        for param in self.exp_emb.parameters():
            param.requires_grad = False

        if self.att_type != "both":
            input_size = emb_dim * 2
        else:
            input_size = emb_dim * 3

        self.vae_encoder = models.classes.MLPEncoder(input_size, hp.mlp_hs, hp.latent_size, hp)
        self.vae_decoder = models.classes.MLPDecoder(hp.latent_size, hp.mlp_hs, emb_dim, hp)

    def forward(self, sent, ind, exp):
        sample_len = len(sent)
        inputs = self.tokenizer(sent, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        max_seq_len = input_tokenized.shape[-1]
        sent_embed = self.text_encoder(input_tokenized, mask)['last_hidden_state']

        inputs = self.get_vae_encoder_input(sent_embed, ind, exp)
        mu_enc, log_var_enc = self.vae_encoder(inputs)
        std = torch.exp(log_var_enc / 2)
        z_dist = Normal(mu_enc, std + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae_decoder(dec_input)

        if self.hp.freeze_decoding == 'False':
            # we train the text decoder independently from the vae
            reconstructed_input_to_decode = reconstructed_input.unsqueeze(1).clone()
            tmp = torch.zeros(sample_len, 1).cuda() + self.tokenizer.bos_token_id
            previous_token = self.text_encoder.embeddings(tmp.long().cuda())
            prev_hidden_state = (
                torch.zeros(self.text_decoder.num_layer, self.hp.b_size, self.text_decoder.hs).type_as(previous_token),
                torch.zeros(self.text_decoder.num_layer, self.hp.b_size, self.text_decoder.hs).type_as(previous_token))

            decoded_tokens = []
            decoder_outputs = []
            for di in range(max_seq_len - 1):
                resized_in_token = self.text_decoder.lin_att_in_para(previous_token)
                extended_enc_rep = reconstructed_input_to_decode.repeat(1, self.max_len, 1)
                tmp = torch.bmm(extended_enc_rep, resized_in_token.transpose(-1, 1))
                attn_weights = masked_softmax(tmp, torch.ones(sample_len, 1).cuda(), 1)
                # assert attn_weights.shape == torch.Size([sample_len, max_seq_len, 1])
                attn_applied = torch.einsum("blh,bld->bdh", extended_enc_rep, attn_weights)
                # output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)
                lstm_input = torch.cat(
                    (reconstructed_input_to_decode, previous_token, attn_applied),
                    -1)
                output_lstm, hidden_state = self.text_decoder.LSTM(lstm_input,
                                                                   prev_hidden_state)
                output_lin = self.text_decoder.lin_lstm_out(output_lstm)
                decoder_output = output_lin
                decoder_outputs.append(decoder_output)
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.text_encoder.embeddings(decoder_tok)
                prev_hidden_state = hidden_state

            decoded_sent = self.tokenizer.batch_decode(torch.stack(decoded_tokens).squeeze(-1).T,
                                                       skip_special_tokens=True)
            resized_outs = torch.stack(decoder_outputs).squeeze(-2).transpose(-1, 1).T
            loss_text_gen = torch.nn.functional.cross_entropy(resized_outs,
                                                              input_tokenized[:, 1:], reduction="sum", ignore_index=1)
            loss_text_gen_reduced = loss_text_gen / sum(sum(mask))
        else:
            loss_text_gen_reduced = 0

        obs_distrib = Normal(reconstructed_input, torch.exp(self.log_scale))
        loss_vae_rec = - torch.sum(obs_distrib.log_prob(sent_embed[:, -1, :]))

        loss_vae_kl = torch.sum(kl_divergence(z_dist, Normal(0, 1)))

        if torch.isnan(loss_vae_kl) or torch.isinf(loss_vae_kl):
            ipdb.set_trace()
        if torch.isnan(loss_vae_rec) or torch.isinf(loss_vae_rec):
            ipdb.set_trace()
        return loss_vae_rec, loss_vae_kl, loss_text_gen_reduced

    def inference(self, sent, ind, exp):
        inputs = self.tokenizer(sent, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        sent_embed = torch.sigmoid(self.text_encoder(input_tokenized, mask)['last_hidden_state'])

        inputs = self.get_vae_encoder_input(sent_embed, ind, exp)
        mu_enc, log_var_enc = self.vae_encoder(inputs)
        std = torch.exp(log_var_enc / 2)
        z_dist = Normal(mu_enc, std + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae_decoder(dec_input)
        ref_dist = Normal(torch.zeros(mu_enc.shape[0], mu_enc.shape[-1]).cuda(),
                          torch.ones(mu_enc.shape[0], mu_enc.shape[-1]).cuda())
        return reconstructed_input, sent_embed[:, -1, :], z_dist, ref_dist

    def get_projection(self, sent, ind, exp):
        inputs = self.tokenizer(sent, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        sent_embed = self.text_encoder(input_tokenized, mask)['last_hidden_state']

        inputs = self.get_vae_encoder_input(sent_embed, ind, exp)
        mu_enc, sig_enc = self.vae_encoder(inputs)
        z_dist = Normal(mu_enc, torch.nn.functional.softplus(sig_enc) + 1e-10)
        dec_input = z_dist.rsample()
        return dec_input

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        sample_len = len(sentences)
        rec, kl, gen = self.forward(sentences, ind_indices, exp_indices)
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
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        sample_len = len(sentences)
        rec, kl, gen = self.forward(sentences, ind_indices, exp_indices)
        val_kl_loss = self.hp.coef_kl * kl / sample_len
        val_rec_loss = self.hp.coef_rec * rec / sample_len
        val_loss = val_rec_loss + val_kl_loss
        self.log('val_rec_loss', val_rec_loss, on_step=False, on_epoch=True)
        self.log('val_kl_loss', val_kl_loss, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_step_outputs):
        if self.hp.plot_latent_space == "True":
            tsne_in_vae_space.main(self.hp, self, self.desc, self.trainer.current_epoch, self.hp.att_type)

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

    def get_vae_encoder_input(self, sent_embed, ind, exp):
        if self.hp.att_type == "both":
            ind_emb = self.ind_emb(ind)
            exp_emb = self.exp_emb(exp)
            inpt = torch.cat([sent_embed[:, -1, :], ind_emb, exp_emb], dim=-1)
        elif self.hp.att_type == "exp":
            exp_emb = self.exp_emb(exp)
            inpt = torch.cat([sent_embed[:, -1, :], exp_emb], dim=-1)
        elif self.hp.att_type == "ind":
            ind_emb = self.ind_emb(ind)
            inpt = torch.cat([sent_embed[:, -1, :], ind_emb], dim=-1)
        else:
            raise Exception(f"Wrong att_type specified. Can be both, exp, or ind. Got: {self.hp.att_type}")
        return inpt
