import ipdb
import argparse
import yaml
import os
import torch
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from models.train_vae import make_xp_title
from models.classes import VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from utils.models import get_latest_model, index_to_one_hot
import matplotlib.lines as mlines


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            xp_title = make_xp_title(args)
            model_name = "/".join(xp_title.split('_'))
            model_path = os.path.join(CFG['modeldir'], model_name)
            model, epoch = load_model(xp_title, model_path, model_name)
            return main(args, model.cuda(), xp_title, epoch)
    else:
        xp_title = make_xp_title(args)
        model_name = "/".join(xp_title.split('_'))
        model_path = os.path.join(CFG['modeldir'], model_name)
        model, epoch = load_model(xp_title, model_path, model_name)
        return main(args, model.cuda(), xp_title, epoch)


def main(args, model, model_name, epoch):
    sub_train = load_sub("train")
    sub_test = load_sub("test")
    train_projections = project_points(sub_train, model, "train")
    test_projections = project_points(sub_test, model, "test")
    trans_points_train, ind_labels_train, exp_labels_train = prep_data_for_viz(train_projections, "train")
    trans_points_test, ind_labels_test, exp_labels_test = prep_data_for_viz(test_projections, "test")
    plot_proj(trans_points_train, ind_labels_train, exp_labels_train,
              trans_points_test, ind_labels_test, exp_labels_test,
              model_name, epoch)
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
    epoch = model_file.split('/')[-1].split('=')[1].split('-')[0]
    return model, epoch


def project_points(data, model, split):
    projections = []
    for i in tqdm(data, desc=f"projecting points of split {split}..."):
        sentence = i["words"]
        ind_index = index_to_one_hot([i["ind_index"]], 20)
        exp_index = index_to_one_hot([i["exp_index"]], 3)
        projection = model.get_projection(sentence, ind_index.cuda(), exp_index.cuda())
        projections.append({'point': projection.detach().cpu().numpy(),
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
    if args.proj_type == "tsne":
        trans_points = fit_transform_by_tsne(points, split)
    elif args.proj_type == "pca":
        trans_points = fit_transform_by_pca(points, split)
    else:
        ipdb.set_trace()
        raise Exception()
    return trans_points, ind_labels, exp_labels


def fit_transform_by_tsne(input_data, split):
    print(f"Fitting TSNE on split {split} for {args.n_tsne} components, {len(input_data)} samples...")
    data_embedded = TSNE(n_components=args.n_comp, n_jobs=-1, verbose=1).fit_transform(np.array(input_data).squeeze(1))
    print(f"TSNE fitted!")
    return data_embedded


def fit_transform_by_pca(input_data, split):
    print(f"Fitting PCA on split {split} for {args.n_tsne} components, {len(input_data)} samples...")
    data_embedded = PCA(n_components=args.n_comp).fit_transform(np.array(input_data).squeeze(1))
    print(f"PCA fitted!")
    return data_embedded


def plot_proj(points_train, inds_train, exps_train, points_test, inds_test, exps_test, model_name, epoch):
    NUM_COLORS = 20
    shape_per_exp, color_legends = get_dicts_for_plot()
    color = cm.rainbow(np.linspace(0, 1, NUM_COLORS))

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"{args.proj_type.upper()} projection of VAE space", fontsize=14)

    ax = fig.add_subplot(211)
    for num, i in enumerate(range(len(points_train))):
        ax.scatter(points_train[num, 0], points_train[num, 1],
                   color=color[inds_train[num]], cmap=color, marker=shape_per_exp[exps_train[num]])
        ax.set_title(f"{args.proj_type.upper()} of {len(points_train)} jobs of split train")
    ax.axis('tight')

    ax = fig.add_subplot(212)
    for num, i in enumerate(range(len(points_test))):
        ax.scatter(points_test[num, 0], points_test[num, 1],
                   color=color[inds_test[num]], cmap=color, marker=shape_per_exp[exps_test[num]])
        ax.set_title(f"{args.proj_type.upper()} of {len(points_test)} jobs of split test")
    ax.axis('tight')

    handles = []
    for k, v in color_legends.items():
        leg = mlines.Line2D([], [], color=color[k], linestyle='None', marker='o',
                            markersize=10, label=v)
        handles.append(leg)
    for k, v in shape_per_exp.items():
        leg = mlines.Line2D([], [], color='black', marker=shape_per_exp[k], linestyle='None',
                            markersize=10, label=v)
        handles.append(leg)

    fig.legend(handles=handles)
    ipdb.set_trace()
    plt.show()
    plt.savefig(f'{args.proj_type}_{model_name}_ep{epoch}.png')


def get_dicts_for_plot():
    shape_per_exp = {0: "x",
                     1: "s",
                     2: 'v'}
    ind_file = "20_industry_dict.pkl"
    with open(os.path.join(CFG["gpudatadir"], ind_file), 'rb') as f_name:
        industry_dict = pkl.load(f_name)
    color_legends = {k: v for k, v in industry_dict.items()}
    return shape_per_exp, color_legends


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--num_point_per_group", type=int, default=5)
    parser.add_argument("--n_comp", type=int, default=2)
    parser.add_argument("--proj_type", type=str, default="pca") # pca or tsne
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
