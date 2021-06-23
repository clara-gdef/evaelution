import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train_vae
from utils import DotDict
from data.visualisation import tsne_in_vae_space

def grid_search(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            main(hparams)
    else:
        main(hparams)


def main(hparams):
    test_results = {}
    dico = init_args(hparams)
    for str_coef_rec in [.2, .4, .6, .8]:
        coef_rec = float(str_coef_rec)
        coef_kl = 1 - coef_rec
        test_results[coef_rec] = {}
        for lr in [1e-2, 1e-3, 1e-4]:
            test_results[coef_rec][lr] = {}
            for b_size in [64, 128, 256]:
                test_results[coef_rec][lr][b_size] = {}
                print(f"Grid Search for (lr={lr}, coef_rec={coef_rec}, batch_size={b_size})")
                dico['lr'] = float(lr)
                dico["coef_rec"] = b_size
                dico["coef_kl"] = coef_kl
                dico["b_size"] = b_size
                arg = DotDict(dico)
                if dico["TRAIN"] == "True":
                    train_vae.init(arg)
                    tsne_in_vae_space.init(hparams)
                if dico["TEST"] == "True":
                    dico["TRAIN"] = "False"
                    test_results[b_size][lr][b_size] = train_vae.init(arg)
                dico["TRAIN"] = "True"
        ## TODO REMOVE THIS - UNINDENT
        res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_" + hparams.model_type)
        with open(res_path, "wb") as f:
            pkl.dump(test_results, f)


def init_args(hparams):
    dico = {"gpus": hparams.gpus,
            "load_dataset": "True",
            "auto_lr_find": "False",
            "load_from_checkpoint": "False",
            "eval_mode": "lastest",
            "model_to_test": "",
            "fine_tune_word_emb": "False",
            "print_preds": "False",
            "optim": hparams.optim,
            "checkpoint": "00",
            "DEBUG": hparams.DEBUG,
            "TEST": hparams.TEST,
            "TRAIN": hparams.TRAIN,
            "subsample": hparams.subsample,
            "num_workers": hparams.num_workers,
            "light": "True",
            "toy_dataset": "False",
            "test_b_size": 1,
            "plot_latent_space": 'False',
            "proj_type": hparams.proj_type,
            "n_comp": hparams.n_comp,
            # model attributes
            "freeze_decoding": "True",
            "b_size": hparams.b_size,
            "max_len": 10,
            "mlp_hs": hparams.mlp_hs,
            "dec_hs": hparams.dec_hs,
            "mlp_layers": hparams.mlp_layers,
            "dec_layers": hparams.dec_layers,
            "model_type": hparams.model_type,
            "scale": 1.,
            # global hyper params
            "exp_levels": hparams.exp_levels,
            "coef_gen": 0.,
            "clip_val": 1.,
            "wd": 0.,
            "dpo": 0.,
            "epochs": hparams.epochs}
    return dico


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_latent_space", type=str, default="True")
    parser.add_argument("--proj_type", type=str, default="pca")
    parser.add_argument("--n_comp", type=int, default=2)
    # model attributes
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--mlp_hs", type=int, default=256)
    parser.add_argument("--dec_hs", type=int, default=768)
    parser.add_argument("--mlp_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="VAE")
    # global hyper params
    parser.add_argument("--coef_gen", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--exp_levels", type=int, default=3)
    hparams = parser.parse_args()
    grid_search(hparams)

