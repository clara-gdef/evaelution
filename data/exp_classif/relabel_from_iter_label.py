import ipdb
import argparse
import yaml
import os
import pickle as pkl
from tqdm import tqdm
from data.datasets import StringIndSubDataset, StringDataset
from utils import get_metrics, handle_fb_preds
from utils.baselines import train_svm, pre_proc_data
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(args)
    else:
        return main(args)


def main(args):
    splits = ["TRAIN", "VALID", "TEST"]
    for split in splits:
        new_ds = load_dataset(args, split)
        with open(os.path.join(CFG["gpudatadir"], f"StringIndSubDataset_3exp_uniform_no_unk_{split}_new2_it30.pkl"), 'rb') as f:
            iterative_ds = pkl.load(f)
        ipdb.set_trace()
        assert len(new_ds) == len(iterative_ds)
        

def load_dataset(hparams, split):
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": hparams.load_dataset,
                 "subsample": hparams.subsample,
                 "max_len": hparams.max_len,
                 "exp_levels": hparams.exp_levels,
                 "rep_file": None,
                 "exp_type": "exp_type",
                 "is_toy": "False"}
    return StringIndSubDataset(**arguments, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample_users", type=int, default=-1)
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--subsample_jobs", type=int, default=-1)
    parser.add_argument("--train_user_len", type=int, default=1000)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--user_step", type=int, default=10)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--f1_threshold", type=int, default=80)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--ind_sub", type=str, default="True")
    parser.add_argument("--initial_check", type=str, default="False")
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--tfidf", type=str, default="True")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
