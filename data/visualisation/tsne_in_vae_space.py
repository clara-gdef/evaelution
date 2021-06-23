import ipdb
import argparse
import yaml
import os
import torch
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from models.train_vae import make_xp_title
from models.classes import VAE
from sklearn.manifold import TSNE
from utils.models import get_latest_model, index_to_one_hot
from cycler import cycler


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            xp_title = make_xp_title(args)
            model_name = "/".join(xp_title.split('_'))
            model_path = os.path.join(CFG['modeldir'], model_name)
            model = load_model(xp_title, model_path, model_name)
            return main(args, model.cuda())
    else:
        xp_title = make_xp_title(args)
        model_name = "/".join(xp_title.split('_'))
        model_path = os.path.join(CFG['modeldir'], model_name)
        model = load_model(xp_title, model_path, model_name)
        return main(args, model.cuda())


def main(args, model):
    sub_train = load_sub("train")
    sub_test = load_sub("test")
    ipdb.set_trace()
    train_projections = project_points(sub_train, model, "train")
    test_projections = project_points(sub_test, model, "test")
    trans_points_train, ind_labels_train, exp_labels_train = prep_data_for_viz(train_projections, "train")
    trans_points_test, ind_labels_test, exp_labels_test = prep_data_for_viz(test_projections, "test")
    ipdb.set_trace()


def load_model(xp_title, model_path, model_name):
    print("Loading model from checkpoint.")
    arguments = {'emb_dim': 768,
                 'hp': args,
                 'desc': xp_title,
                 "num_ind": 20,
                 "model_path": model_path,
                 "num_exp_level": 3,
                 "datadir": CFG["gpudatadir"]}
    print("Initiating model...")
    model = VAE(**arguments)
    print("Model Loaded.")
    model_file = get_latest_model(CFG["modeldir"], model_name)
    try:
        model.load_state_dict(torch.load(model_file)["state_dict"])
    except RuntimeError:
        model.load_state_dict(torch.load(model_file))
    print(f"Model loaded from checkpoint: {model_file}")
    return model


def project_points(data, model, split):
    projections = []
    for i in tqdm(data, desc=f"projecting points of split {split}..."):
        sentence = i["words"]
        ind_index = index_to_one_hot(index_to_one_hot([i["ind_index"]], 20), 20)
        exp_index = index_to_one_hot([i["exp_index"]], 3)
        projection = model.get_projection(sentence, ind_index, exp_index)
        projections.append({'point': projection,
                            "ind_index": i["ind_index"],
                            "exp_index": i["exp_index"]})
    return projections


def load_sub(split):
    tgt_file = os.path.join(CFG["gpudatadir"], f"viz_subgroup_{split}.pkl")
    with open(tgt_file, 'rb') as f:
        sub = pkl.load(f)
    return sub


def prep_data_for_viz(data_dict, split):
    points = [i["point"] for i in data_dict]
    ind_labels = [i["ind_index"] for i in data_dict]
    exp_labels = [i["exp_index"] for i in data_dict]
    trans_points = fit_transform_by_tsne(points, split)
    return trans_points, ind_labels, exp_labels


def fit_transform_by_tsne(input_data, split):
    print(f"Fitting TSNE on split {split} for {args.n_tsne} components, {len(input_data)} samples...")
    data_embedded = TSNE(n_components=args.n_tsne).fit_transform(input_data)
    print(f"TSNE fitted!")
    return data_embedded


# def plot_tsne(points, inds, exps, split):
    # NUM_COLORS = 20
    #
    # cm = plt.get_cmap('gist_rainbow')
    # fig = plt.figure(figsize=(15, 8))
    # fig.suptitle(f"TSNE of {len(points)} jobs of split {split}", fontsize=14)
    # ax = fig.add_subplot(211)
    #
    # ax.scatter(points[:, 0], points[:, 1],
    #            c=val_set_labels[:args.subsample], cmap=plt.cm.Spectral)
    # ax.set_title("PCA over %s points (%.2g sec)" % (args.subsample, t1 - t0))
    # ax.axis('tight')
    #
    # ax = fig.add_subplot(212)
    #
    # ax.scatter(val_set_embedded_tsne[:args.subsample, 0], val_set_embedded_tsne[:, 1],
    #            c=val_set_labels[:args.subsample], cmap=plt.cm.Spectral)
    # ax.set_title("TSNE over %s points (%.2g sec)" % (args.subsample, t3 - t2))
    # ax.axis('tight')
    # plt.savefig('tsne_pca_' + str(args.subsample) + "_" + args.ft_type + "_" + str(iteration) + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--num_point_per_group", type=int, default=5)
    parser.add_argument("--n_tsne", type=int, default=2)
    # model attributes
    parser.add_argument("--freeze_decoding", type=str, default="True")
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--mlp_hs", type=int, default=256)
    parser.add_argument("--dec_hs", type=int, default=768)
    parser.add_argument("--mlp_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="vae_no_dec")
    parser.add_argument("--optim", default="adam")
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_levels", type=int, default=3)

    args = parser.parse_args()
    init(args)
