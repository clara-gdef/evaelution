import ipdb
import torch
import os
import glob
from sklearn.metrics import f1_score, accuracy_score


def collate_for_VAE(batch):
    jobs = [i[0] for i in batch]
    ind = [i[1] for i in batch]
    exp = [i[2] for i in batch]
    return jobs, index_to_one_hot(ind, 20), index_to_one_hot(exp, 3)


def index_to_one_hot(indices, max_features):
    tnsr = torch.zeros(len(indices), max_features)
    for num, i in enumerate(indices):
        tnsr[num, i] = 1.
    return tnsr


def get_latest_model(modeldir, model_name):
    model_path = os.path.join(modeldir, model_name)
    model_files = glob.glob(os.path.join(model_path, "*.ckpt"))
    latest_file = max(model_files, key=os.path.getctime)
    return latest_file


def masked_softmax(logits, mask, seq_len):
    sample_len = logits.shape[0]
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    mask = mask.type(dtype=logits.dtype)
    if seq_len > 1:# we compare the encoder output against the whole sequence
        weights = torch.exp(logits) * mask.unsqueeze(-1).expand(sample_len, seq_len, seq_len-1)
    # we compare the encoder output against only 1 token
    else:
        weights = torch.exp(logits) * mask.unsqueeze(-1)
    denominator = 1e-7 + torch.sum(weights, dim=1, keepdim=True)
    return weights / denominator


def get_metrics(preds, labels, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def handle_fb_preds(pred):
    return [int(float(i.split("__label__")[-1])) for i in pred[0]]