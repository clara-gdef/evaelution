import os
import ipdb
import argparse
import yaml
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.datasets.StringIndSubDataset import StringIndSubDataset
from models.classes.VAE import VAE
from utils.models import collate_for_VAE, get_latest_model, collate_for_VAE_exp, collate_for_VAE_ind, collate_for_VAE_no_att


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
    collate_fn, num_ind, num_exp = get_collate_fn_and_class_nums(hparams)

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
        datasets = load_datasets(CFG, hparams, ["TRAIN", "TRAIN"])
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
        print("Dataloaders initiated.")
    print("Dataloaders initiated.")
    arguments = {'emb_dim': 768,
                 'hp': hparams,
                 'desc': xp_title,
                 "num_ind": num_ind,
                 "model_path": model_path,
                 "num_exp_level": num_exp,
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
            trainer.current_epoch = int(str(hparams.checkpoint).split("=")[0].split("-")[0])
            print("Resuming training from checkpoint : " + model_file + ".")
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
            raise Exception(f"Wrong argument provided for EVAL_MODE, can be either \"latest\" or \"spe\", {hparams.eval_mode} was given.")
        if hparams.TRAIN == "False":
            print("Loading model to evaluate...")
            try:
                model.load_state_dict(torch.load(model_file)["state_dict"])
            except KeyError:
                model.load_state_dict(torch.load(model_file))
        print(f"Evaluating model : {model_file}.")
        datasets = load_datasets(CFG, hparams, ["TEST"], hparams.load_dataset)
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=hparams.test_b_size, collate_fn=collate_fn,
                                 num_workers=hparams.num_workers, drop_last=True)
        trainer.test(test_dataloaders=test_loader, model=model)
        return model.metrics


def load_datasets(CFG, hparams, splits):
    datasets = []
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": hparams.load_dataset,
                 "subsample": hparams.subsample,
                 "max_len": hparams.max_len,
                 "exp_levels": hparams.exp_levels,
                 "rep_file": None,
                 "exp_type": "uniform",
                 "suffix": hparams.suffix,
                 "is_toy": hparams.toy_dataset}
    for split in splits:
        datasets.append(StringIndSubDataset(**arguments, split=split))
    return datasets


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


def get_collate_fn_and_class_nums(hparams):
    if hparams.att_type == "both":
        return collate_for_VAE, 20, 3
    elif hparams.att_type == "exp":
        return collate_for_VAE_exp, 0, 3
    elif hparams.att_type == "ind":
        return collate_for_VAE_ind, 20, 0
    elif hparams.att_type == "none":
        return collate_for_VAE_no_att, 0, 0

    else:
        raise Exception(f"Wrong att_type specified. Can be exp, ind, both or none. Got: {hparams.att_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--suffix", type=str, default="") # "" or "_sub_by_subspace"
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--eval_mode", type=str, default="latest") # can be "spe" or "latest"
    parser.add_argument("--model_to_test", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="01")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--test_b_size", type=int, default=1)
    parser.add_argument("--clip_val", type=float, default=1.)
    parser.add_argument("--kl_ep_threshold", type=int, default=10)
    parser.add_argument("--plot_latent_space", type=str, default="True")
    parser.add_argument("--plot_grad", type=str, default="False")
    parser.add_argument("--proj_type", type=str, default="tsne")
    parser.add_argument("--n_comp", type=int, default=2)
    parser.add_argument("--att_type", type=str, default="ind") # can be both, exp, ind or "none"
    # model attributes
    parser.add_argument("--freeze_decoding", type=str, default="True")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--b_size", type=int, default=256)
    parser.add_argument("--mlp_hs", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--logscale", type=float, default=1.)
    parser.add_argument("--model_type", type=str, default="VAEnosigmoid")
    # global hyper params
    parser.add_argument("--coef_rec", type=float, default=.5)
    parser.add_argument("--coef_kl", type=float, default=.5)
    parser.add_argument("--coef_gen", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_levels", type=int, default=3)
    hparams = parser.parse_args()
    init(hparams)
