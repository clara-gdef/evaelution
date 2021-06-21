import ipdb
import torch


def collate_for_VAE(batch):
    jobs = [i[0] for i in batch]
    ind = [i[1] for i in batch]
    exp = [i[2] for i in batch]
    ipdb.set_trace()

    return jobs, index_to_one_hot(ind, 20), index_to_one_hot(exp, 3)


def index_to_one_hot(indices, max_features):
    tnsr = torch.zeros(len(indices), max_features)
    for num, i in enumerate(indices):
        tnsr[num, i] = 1.
    return tnsr
