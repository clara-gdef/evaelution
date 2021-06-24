import ipdb
import torch
import pytorch_lightning as pl
from utils.models import masked_softmax
import models.classes
from data.visualisation import tsne_in_vae_space
from torch.distributions.normal import Normal
from transformers import CamembertTokenizer, CamembertModel


class LatentSpaceDec(pl.LightningModule):
    def __init__(self, datadir, emb_dim, desc, model_path, num_ind, num_exp_level, vae_model, epoch, hp):
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
        self.epoch = epoch

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.voc_size = self.tokenizer.vocab_size

        self.text_encoder = CamembertModel.from_pretrained('camembert-base')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.vae = vae_model
        for param in self.vae.parameters():
            param.requires_grad = False

        self.text_decoder = models.classes.LSTMAttentionDecoder(self.hp.dec_hs, self.emb_dim, self.voc_size, hp)

    def forward(self, sent, ind, exp, batch_nb):
        sample_len = len(ind)
        inputs = self.tokenizer(sent, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        max_seq_len = input_tokenized.shape[-1]
        sent_embed = self.text_encoder(input_tokenized, mask)['last_hidden_state']

        inputs = torch.cat([sent_embed[:, -1, :], ind, exp], dim=-1)
        params = self.vae.vae_encoder(inputs)
        mu_enc, sig_enc = params[:, :self.hp.mlp_hs], params[:, self.hp.mlp_hs:]
        z_dist = Normal(mu_enc, torch.nn.functional.softplus(sig_enc) + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae.vae_decoder(dec_input)

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
            attn_applied = torch.einsum("blh,bld->bdh", extended_enc_rep, attn_weights)
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

        if self.hp.print_preds == "True" and batch_nb == 0:
            decoded_sent = self.tokenizer.decode(torch.stack(decoded_tokens).squeeze(-1).T[0],
                                                       skip_special_tokens=True)
            print("DECODED SENTENCE: " + decoded_sent)
            print("EXPECTED SENTENCE: " + sent[0])

        resized_outs = torch.stack(decoder_outputs).squeeze(-2).transpose(-1, 1).T
        loss_text_gen = torch.nn.functional.cross_entropy(resized_outs,
                                                          input_tokenized[:, 1:], reduction="sum", ignore_index=1)
        loss_text_gen_reduced = loss_text_gen / sum(sum(mask))
        return loss_text_gen_reduced

    def inference(self, sent, ind, exp):
        sample_len = len(ind)
        inputs = self.tokenizer(sent, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        max_seq_len = input_tokenized.shape[-1]

        sent_embed = self.text_encoder(input_tokenized, mask)['last_hidden_state']

        inputs = torch.cat([sent_embed[:, -1, :], ind, exp], dim=-1)
        params = self.vae_encoder(inputs)
        mu_enc, sig_enc = params[:, :self.hp.mlp_hs], params[:, self.hp.mlp_hs:]
        z_dist = Normal(mu_enc, torch.nn.functional.softplus(sig_enc) + 1e-10)
        dec_input = z_dist.rsample()
        reconstructed_input = self.vae_decoder(dec_input)

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
            attn_applied = torch.einsum("blh,bld->bdh", extended_enc_rep, attn_weights)
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

        resized_outs = torch.stack(decoder_outputs).squeeze(-2).transpose(-1, 1).T
        return resized_outs

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        loss = self.forward(sentences, ind_indices, exp_indices, batch_nb)
        self.log('train_gen_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        val_loss = self.forward(sentences, ind_indices, exp_indices, batch_nb)
        self.log('val_loss', val_loss)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.epoch += 1
        if self.hp.plot_latent_space == "True":
            tsne_in_vae_space.main(self.hp, self, self.desc, self.epoch)

    def test_epoch_start(self):
        ipdb.set_trace()

    def test_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        decoded_sentence = self.inference(sentences, ind_indices, exp_indices)
        ipdb.set_trace()
        return {"ppl": loss_vae_kl, "bleu_score": loss_vae_rec}

    def test_epoch_end(self):
        ipdb.set_trace()
