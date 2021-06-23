import ipdb
import torch
import pytorch_lightning as pl
from utils.models import masked_softmax
import models.classes
from models.classes import MLP, LSTMAttentionDecoder
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from transformers import CamembertTokenizer, CamembertModel
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MBartModel


class VAE(pl.LightningModule):
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

        # self.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        # self.tokenizer.src_lang = "fr_XX"
        #
        # tmp = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        # self.text_encoder = tmp.model.encoder
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        # self.text_decoder = MBartModel.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        # self.text_decoder.is_decoder = True

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.voc_size = self.tokenizer.vocab_size

        self.text_encoder = CamembertModel.from_pretrained('camembert-base')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if self.hp.freeze_decoding == 'False':
            self.text_decoder = models.classes.LSTMAttentionDecoder(self.hp.dec_hs, self.emb_dim, self.voc_size, hp)

        input_size = emb_dim + num_ind + num_exp_level
        output_size = 2 * hp.mlp_hs

        self.vae_encoder = MLP(input_size, output_size, hp.mlp_layers, hp)
        self.vae_decoder = MLP(hp.mlp_hs, emb_dim, hp.mlp_layers, hp)

    def forward(self, sent, ind, exp):
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

        # we train the text decoder independently from the vae
        reconstructed_input_to_decode = reconstructed_input.unsqueeze(1).clone()
        tmp = torch.zeros(sample_len, 1).cuda() + self.tokenizer.bos_token_id
        previous_token = self.text_encoder.embeddings(tmp.long().cuda())
        prev_hidden_state = (
            torch.zeros(self.text_decoder.num_layer, self.hp.b_size, self.text_decoder.hs).type_as(previous_token),
            torch.zeros(self.text_decoder.num_layer, self.hp.b_size, self.text_decoder.hs).type_as(previous_token))

        # prev_hidden_state = (reconstructed_input_to_decode.repeat(1, self.hp.dec_layers, 1).cuda(),
        #                      reconstructed_input_to_decode.repeat(1, self.hp.dec_layers, 1).cuda())

        if self.hp.freeze_decoding == 'False':
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

        # TODO when you're older, use the NLL
        loss_vae_rec = torch.nn.functional.mse_loss(sent_embed[:, -1, :], reconstructed_input)

        ref_dist = Normal(0, 1)
        loss_vae_kl = torch.mean(kl_divergence(z_dist, ref_dist))

        return loss_vae_rec, loss_vae_kl, loss_text_gen_reduced

    def inference(self, ):
        ipdb.set_trace()

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        rec, kl, gen = self.forward(sentences, ind_indices, exp_indices)
        loss = rec + kl + gen
        self.log('train_rec_loss', rec)
        self.log('train_kl_loss', kl)
        self.log('train_gen_loss', gen)
        self.log('train_loss', loss)
        return {"loss": loss}
    #
    # def training_step_end(self, training_step_outputs):
    #     return {'loss': training_step_outputs['loss'].mean()}

    def validation_step(self, batch, batch_nb):
        sentences, ind_indices, exp_indices = batch[0], batch[1], batch[2]
        rec, kl, gen = self.forward(sentences, ind_indices, exp_indices)
        val_loss = rec + kl + gen
        self.log('val_rec_loss', rec)
        self.log('val_kl_loss', kl)
        self.log('val_gen_loss', gen)
        self.log('val_loss', val_loss)
        return {"val_loss": val_loss}

    def test_epoch_start(self):
        ipdb.set_trace()

    def test_step(self):
        ipdb.set_trace()

    def test_epoch_end(self):
        ipdb.set_trace()
