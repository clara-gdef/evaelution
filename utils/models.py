import ipdb
import torch
import os
import glob
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def collate_for_VAE(batch):
    jobs = [i[0] for i in batch]
    ind = [int(i[1]) for i in batch]
    exp = [int(i[2]) for i in batch]
    return jobs, index_to_one_hot(ind, 20), index_to_one_hot(exp, 3)


def collate_for_VAE_exp(batch):
    jobs = [i[0] for i in batch]
    exp = [int(i[2]) for i in batch]
    return jobs, None, index_to_one_hot(exp, 3)


def collate_for_VAE_ind(batch):
    jobs = [i[0] for i in batch]
    ind = [int(i[1]) for i in batch]
    return jobs, index_to_one_hot(ind, 20), None


def collate_for_VAE_emb(batch):
    jobs = [i[0] for i in batch]
    ind = [int(i[1]) for i in batch]
    exp = [int(i[2]) for i in batch]
    return jobs, torch.LongTensor(ind), torch.LongTensor(exp)


def collate_for_VAE_emb_exp(batch):
    jobs = [i[0] for i in batch]
    exp = [int(i[2]) for i in batch]
    return jobs, None, torch.LongTensor(exp)


def collate_for_VAE_emb_ind(batch):
    jobs = [i[0] for i in batch]
    ind = [int(i[1]) for i in batch]
    return jobs, torch.LongTensor(ind), None


def collate_for_VAE_no_att(batch):
    jobs = [i[0] for i in batch]
    ind = [int(i[1]) for i in batch]
    return jobs, index_to_one_hot(ind, 20), None


def collate_for_VAE_mnist(batch):
    images = [i[0] for i in batch]
    labels = [int(i[1]) for i in batch]
    return torch.stack(images), index_to_one_hot(labels, 10)


def index_to_one_hot(indices, max_features):
    tnsr = torch.zeros(len(indices), max_features)
    for num, i in enumerate(indices):
        tnsr[num, int(i)] = 1.
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


def plot_grad_flow(named_parameters, desc):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for name, param in named_parameters:
        if (param.requires_grad) and ("bias" not in name):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean())
            max_grads.append(param.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), torch.stack(max_grads).cpu().numpy(), alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), torch.stack(ave_grads).cpu().numpy(), alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f"img/plot_grad_flow_{desc}.png")


def get_metrics_at_k(predictions, labels, num_classes, handle):
    out_predictions = []
    for index, pred in enumerate(predictions):
        if labels[index] in pred:
            if (type(labels[index]) == torch.Tensor) or (type(labels[index]) == np.ndarray):
                out_predictions.append(labels[index].item())
            else:
                out_predictions.append(labels[index])
        else:
            if (type(pred[0]) == torch.Tensor) or (type(pred[0]) == np.ndarray):
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle)
