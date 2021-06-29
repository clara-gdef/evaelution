import argparse
import os
import pickle as pkl
import urllib

import ipdb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from tqdm import tqdm

import models
from utils.models import get_latest_model, index_to_one_hot


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            xp_title = make_xp_title(args)
            model_name = "/".join(xp_title.split('_'))
            model_path = os.path.join(CFG['modeldir'], model_name)
            model, epoch = load_model(args, xp_title, model_path, model_name, args.att_type)
            return main(args, model.cuda(), xp_title, epoch, args.att_type)
    else:
        xp_title = make_xp_title(args)
        model_name = "/".join(xp_title.split('_'))
        model_path = os.path.join(CFG['modeldir'], model_name)
        model, epoch = load_model(args, xp_title, model_path, model_name, args.att_type)
        return main(args, model.cuda(), xp_title, epoch, args.att_type)


def main(args, model, model_name, epoch, att_type):
    sub_train = load_sub("train", att_type)
    sub_test = load_sub("test", att_type)
    train_projections = project_points(sub_train, model, "train", att_type)
    test_projections = project_points(sub_test, model, "test", att_type)
    trans_points_train, ind_labels_train, exp_labels_train = prep_data_for_viz(args, train_projections, "train", att_type)
    trans_points_test, ind_labels_test, exp_labels_test = prep_data_for_viz(args, test_projections, "test", att_type)
    plot_proj(args, trans_points_train, ind_labels_train, exp_labels_train,
              trans_points_test, ind_labels_test, exp_labels_test,
              model_name, epoch, att_type)


def load_model(args, xp_title, model_path, model_name, att_type):
    print("Loading model from checkpoint.")
    if att_type == "mnist":
        arguments = {'emb_dim': 784,
                     'hp': args,
                     'desc': xp_title,
                     "num_classes": 10,  # corresponds to classes
                     "model_path": model_path,
                     "datadir": CFG["gpudatadir"]
                     }
        print("Initiating model...")
        model = models.classes.VAEMnist(**arguments)
    else:
        num_ind, num_exp = get_collate_fn_and_class_nums(args)

        arguments = {'emb_dim': 768,
                     'hp': args,
                     'desc': xp_title,
                     "num_ind": num_ind,
                     "model_path": model_path,
                     "num_exp_level": num_exp,
                     "datadir": CFG["gpudatadir"]}
        print("Initiating model...")
        model = models.classes.VAE(**arguments)
    print("Model Loaded.")
    model_file = get_latest_model(CFG["modeldir"], model_name)
    try:
        model.load_state_dict(torch.load(model_file)["state_dict"])
    except RuntimeError:
        model.load_state_dict(torch.load(model_file))
    print(f"Model loaded from checkpoint: {model_file}")
    epoch = model_file.split('/')[-1].split('=')[1].split('-')[0]
    return model, epoch


def project_points(data, model, split, att_type):
    projections = []
    cnt = 0
    for i in tqdm(data, desc=f"projecting points of split {split}..."):
        if att_type == "mnist":
            if cnt < 300:
                images = i[0].view(1, -1)
                labels = index_to_one_hot([i[1]], 10)
                projection = model.get_projection(images.cuda(), labels.cuda())
                projections.append({'point': projection.detach().cpu().numpy(),
                                    "label": i[1]})
                cnt += 1
        else:
            sentence = i["words"]
            ind_index = index_to_one_hot([i["ind_index"]], 20)
            exp_index = index_to_one_hot([i["exp_index"]], 3)
            projection = model.get_projection(sentence, ind_index.cuda(), exp_index.cuda())
            projections.append({'point': projection.detach().cpu().numpy(),
                                "ind_index": i["ind_index"],
                                "exp_index": i["exp_index"]})
    return projections


def load_sub(split, att_type):
    if att_type == "mnist":
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        sub = torchvision.datasets.MNIST("/local/gainondefor/work/data/evaelution",
                                         train=(split == "train"), transform=torchvision.transforms.ToTensor(),
                                         download=False)
    else:
        tgt_file = f"/local/gainondefor/work/data/evaelution/viz_subgroup_{split}.pkl"
        with open(tgt_file, 'rb') as f:
            sub = pkl.load(f)
    return sub


def prep_data_for_viz(args, data_dict, split, att_type):
    points = [i["point"] for i in data_dict]
    if args.proj_type == "tsne":
        trans_points = fit_transform_by_tsne(args, points, split)
    elif args.proj_type == "pca":
        trans_points = fit_transform_by_pca(args, points, split)
    else:
        ipdb.set_trace()
        raise Exception()
    if att_type == "mnist":
        labels = [i["label"] for i in data_dict]
        return trans_points, labels, [0] * len(labels)
    else:
        ind_labels = [i["ind_index"] for i in data_dict]
        exp_labels = [i["exp_index"] for i in data_dict]
        return trans_points, ind_labels, exp_labels


def fit_transform_by_tsne(args, input_data, split):
    print(f"Fitting TSNE on split {split} for {args.n_comp} components, {len(input_data)} samples...")
    normed_data = normalize(np.array(input_data).squeeze(1))
    data_embedded = TSNE(n_components=args.n_comp, n_jobs=-1, verbose=1).fit_transform(normed_data)
    print(f"TSNE fitted!")
    return data_embedded


def fit_transform_by_pca(args, input_data, split):
    print(f"Fitting PCA on split {split} for {args.n_comp} components, {len(input_data)} samples...")
    normed_data = normalize(np.array(input_data).squeeze(1))
    data_embedded = PCA(n_components=args.n_comp).fit_transform(normed_data)
    print(f"PCA fitted!")
    return data_embedded


def plot_proj(args, points_train, inds_train, exps_train, points_test, inds_test, exps_test, model_name, epoch,
              att_type):
    print("Initiating dicts and lists for colors...")
    shape_per_exp, color_legends, color = get_dicts_for_plot(att_type)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"{args.proj_type.upper()} projection of VAE space, att={att_type}, epoch {epoch}", fontsize=14)
    print("Scattering points of train split...")
    ax = fig.add_subplot(211)
    for num, i in enumerate(range(len(points_train))):
        ax.scatter(points_train[num, 0], points_train[num, 1],
                   color=color[inds_train[num]], cmap=color, marker=shape_per_exp[exps_train[num]])
        ax.set_title(f"{args.proj_type.upper()} of {len(points_train)} jobs of split train")
    ax.axis('tight')

    print("Scattering points of test split...")
    ax = fig.add_subplot(212)
    for num, i in enumerate(range(len(points_test))):
        ax.scatter(points_test[num, 0], points_test[num, 1],
                   color=color[inds_test[num]], cmap=color, marker=shape_per_exp[exps_test[num]])
        ax.set_title(f"{args.proj_type.upper()} of {len(points_test)} jobs of split test")
    ax.axis('tight')

    print("Building legends for markers and colors...")
    handles = []
    if color_legends is not None:
        for k, v in tqdm(color_legends.items(), desc="Building legends for colors..."):
            leg = mlines.Line2D([], [], color=color[k], linestyle='None', marker='o',
                                markersize=10, label=v)
            handles.append(leg)
    tmp = [v for v in shape_per_exp.values()]
    if tmp != ["x", "x", "x"]:
        for k, v in tqdm(shape_per_exp.items(), desc="Building legends for markers..."):
            leg = mlines.Line2D([], [], color='black', marker=shape_per_exp[k], linestyle='None',
                                markersize=10, label=f'e{k}')
            handles.append(leg)
    fig.legend(handles=handles)
    print("Legends for markers and colors done.")
    # plt.show()
    dest_file = f'img/{args.proj_type}_{model_name}_ep{epoch}.png'
    print(f"Saving picture at {dest_file}...")
    plt.savefig(dest_file)
    print(f"Figure saved at {dest_file}")
    plt.close()


def get_dicts_for_plot(att_type):
    if att_type == "both":
        shape_per_exp = {0: "x",
                         1: "s",
                         2: 'v'}
        ind_file = "/local/gainondefor/work/data/evaelution/20_industry_dict.pkl"
        with open(ind_file, 'rb') as f_name:
            industry_dict = pkl.load(f_name)
        color_legends = {k: v for k, v in industry_dict.items()}
        color = cm.rainbow(np.linspace(0, 1, 20))
    elif att_type == "exp":
        shape_per_exp = {0: "x",
                         1: "s",
                         2: 'v'}
        color_legends = None
        unique_color = [0.5, 0., 1., 1.]
        color = np.array([unique_color] * 20)
    elif att_type == "ind":
        shape_per_exp = {0: "x",
                         1: "x",
                         2: 'x'}
        ind_file = "/local/gainondefor/work/data/evaelution/20_industry_dict.pkl"
        with open(ind_file, 'rb') as f_name:
            industry_dict = pkl.load(f_name)
        color_legends = {k: v for k, v in industry_dict.items()}
        color = cm.rainbow(np.linspace(0, 1, 20))
    elif att_type == "mnist":
        shape_per_exp = {0: "x",
                         1: "x",
                         2: 'x'}
        color_legends = {k: str(k) for k in range(10)}
        color = cm.rainbow(np.linspace(0, 1, 10))
    elif att_type == "none":
        shape_per_exp = {0: "x",
                         1: "x",
                         2: 'x'}
        color_legends = None
        unique_color = [0.5, 0., 1., 1.]
        color = np.array([unique_color] * 20)
    else:
        raise Exception(f"Wrong att_type specified. Can be exp, ind, both or none. Got: {att_type}")
    return shape_per_exp, color_legends, color


def make_xp_title(hparams):
    xp_title = f"{hparams.model_type}_bs{hparams.b_size}_mlphs{hparams.mlp_hs}_lr{hparams.lr}_{hparams.optim}"
    if hparams.coef_rec != .5:
        xp_title += f"_coef_rec{hparams.coef_rec}"
    if hparams.att_type != "both":
        xp_title += f"_{hparams.att_type}Only"
    if hparams.subsample != -1:
        xp_title += f"sub{hparams.subsample}"
    print("xp_title = " + xp_title)
    return xp_title


def get_collate_fn_and_class_nums(hparams):
    if hparams.att_type == "both":
        return 20, 3
    elif hparams.att_type == "exp":
        return 0, 3
    elif hparams.att_type == "ind":
        return 20, 0
    elif hparams.att_type == "none":
        return 0, 0
    else:
        raise Exception(f"Wrong att_type specified. Can be exp, ind, both or none. Got: {hparams.att_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--num_point_per_group", type=int, default=5)
    parser.add_argument("--n_comp", type=int, default=2)
    parser.add_argument("--proj_type", type=str, default="tsne")  # pca or tsne
    parser.add_argument("--att_type", type=str, default="ind")  # both or exp or ind
    # model attributes
    parser.add_argument("--freeze_decoding", type=str, default="True")
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--mlp_hs", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="VAEnosigmoid")
    parser.add_argument("--optim", default="adam")
    # global hyper params
    parser.add_argument("--coef_rec", type=float, default=.5)
    parser.add_argument("--logscale", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
