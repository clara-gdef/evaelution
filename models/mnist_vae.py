import argparse
import os

import ipdb
from six.moves import urllib
import pytorch_lightning as pl
import torch
import torchvision
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classes.VAE import VAE
from utils.models import get_latest_model


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(hparams)
    else:
        return main(hparams)


def main(hparams):
    xp_title = make_xp_title(hparams)
    model_name = "/".join(xp_title.split('_'))
    logger, checkpoint_callback, early_stop_callback, model_path = init_lightning(CFG, xp_title, model_name)
    call_back_list = [checkpoint_callback, early_stop_callback]

    if hparams.DEBUG == "True":
        trainer = pl.Trainer(gpus=1,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             gradient_clip_val=hparams.clip_val
                             )
        num_workers = 0
    else:
        trainer = pl.Trainer(gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             accelerator='ddp_spawn',
                             gradient_clip_val=hparams.clip_val
                             )
        num_workers = hparams.num_workers
    if hparams.TRAIN == "True":
        # todo : remove after debug
        datasets = load_datasets(CFG, hparams)
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size,
                                  num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
        print("Dataloaders initiated.")
    print("Dataloaders initiated.")
    arguments = {'emb_dim': 784,
                 'hp': hparams,
                 'desc': xp_title,
                 "num_ind": 0,
                 "model_path": model_path,
                 "num_exp_level": 0,
                 "datadir": CFG["gpudatadir"]
                 }
    print("Initiating model...")
    model = VAE(**arguments)
    print("Model Loaded.")
    if hparams.TRAIN == "True":
        if hparams.load_from_checkpoint == "True":
            print("Loading from previous checkpoint...")
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
            model.load_state_dict(torch.load(model_file)["state_dict"])
            print("Resuming training from checkpoint : " + model_file + ".")
            if hparams.divide_lr > 0:
                new_lr = hparams.lr / hparams.divide_lr
                model.hp.lr = new_lr
                print(f"lr updated from {new_lr * hparams.divide_lr} to {new_lr}")
        if hparams.auto_lr_find == "True":
            print("looking for best lr...")
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

            # Results can be found in
            print(lr_finder.results)

            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()

            # update hparams of the model
            model.hp.lr = new_lr
            ipdb.set_trace()
        print("Starting training for " + xp_title + "...")

        return trainer.fit(model.cuda(), train_loader, valid_loader)
    if hparams.TEST == "True":
        if hparams.eval_mode == "latest":
            model_file = get_latest_model(CFG["modeldir"], model_name)
        elif hparams.eval_mode == "spe":
            model_file = os.path.join(CFG["modeldir"], model_name + "/" + hparams.model_to_test)
        else:
            raise Exception(
                f"Wrong argument provided for EVAL_MODE, can be either \"latest\" or \"spe\", {hparams.eval_mode} was given.")
        if hparams.TRAIN == "False":
            print("Loading model to evaluate...")
            try:
                model.load_state_dict(torch.load(model_file)["state_dict"])
            except KeyError:
                model.load_state_dict(torch.load(model_file))
        print(f"Evaluating model : {model_file}.")
        datasets = load_datasets(CFG, hparams, ["TEST"], hparams.load_dataset)
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=hparams.test_b_size,
                                 num_workers=hparams.num_workers, drop_last=True)
        trainer.test(test_dataloaders=test_loader, model=model)
        return model.metrics


def load_datasets(CFG, hparams):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    dataset_train = torchvision.datasets.MNIST(CFG["datadir"],
                                               train=True, transform=transforms.ToTensor(),
                                               download=(hparams.load_dataset == "False"))
    dataset_test = torchvision.datasets.MNIST(CFG["datadir"],
                                              train=False, transform=transforms.ToTensor(),
                                              download=(hparams.load_dataset == "False"))
    ipdb.set_trace()
    return dataset_train, dataset_test


def init_lightning(CFG, xp_title, model_name):
    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    print("callback initiated.")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback, model_path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--suffix", type=str, default="")  # "" or "_sub_by_subspace"
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--eval_mode", type=str, default="latest")  # can be "spe" or "latest"
    parser.add_argument("--model_to_test", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="01")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--test_b_size", type=int, default=1)
    parser.add_argument("--clip_val", type=float, default=.5)
    parser.add_argument("--plot_latent_space", type=str, default="False")
    parser.add_argument("--plot_grad", type=str, default="False")
    parser.add_argument("--proj_type", type=str, default="tsne")
    parser.add_argument("--n_comp", type=int, default=2)
    parser.add_argument("--att_type", type=str, default="both")  # can be both, exp or ind
    # model attributes
    parser.add_argument("--freeze_decoding", type=str, default="True")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--mlp_hs", type=int, default=200)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--logscale", type=float, default=1.)
    parser.add_argument("--model_type", type=str, default="mnist_vae")
    # global hyper params
    parser.add_argument("--coef_rec", type=float, default=.5)
    parser.add_argument("--coef_kl", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_levels", type=int, default=3)
    hparams = parser.parse_args()
    init(hparams)
