import argparse
import joblib
import yaml
import os
import ipdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import Counter
from data.datasets.StringIndSubDataset import StringIndSubDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        data_train, data_valid, data_test = get_data(CFG, args)

        train_lu = data_train.user_lookup
        valid_lu = data_valid.user_lookup
        test_lu = data_test.user_lookup
        exp_seq, carreer_len = Counter(), Counter()
        exp_seq, carreer_len = get_exp_sequence(train_lu, data_train.tuples, exp_seq, carreer_len, "train")
        exp_seq, carreer_len = get_exp_sequence(valid_lu, data_valid.tuples, exp_seq, carreer_len, "valid")
        exp_seq, carreer_len = get_exp_sequence(test_lu, data_test.tuples, exp_seq, carreer_len, "test")
        total_users = len(train_lu)+len(valid_lu)+len(test_lu)
        exp_seq.most_common(10)
        prct_exp_seq = [(i, 100*v/total_users) for i, v in exp_seq.most_common(10)]
        prct_career_len = [(i, 100*v/total_users) for i, v in carreer_len.most_common(10)]
        ipdb.set_trace()


def get_exp_sequence(users, jobs, exp_seq, carreer_len, split):
    for num, user in enumerate(tqdm(users, desc=f"parsing users for {split} split...")):
        current_user = users[user]
        start = current_user[0]
        end = current_user[1]
        current_seq = []
        for job in range(start, end):
            if split == "test":
                tmp = jobs[job]["exp_index"]
            else:
                tmp = jobs[job][-1]
            current_seq.append(tmp)
        if len(current_seq) > 0:
            carreer_len[len(current_seq)] += 1
            exp_seq[str(current_seq)] += 1
    print(f"Number of sequence acquired: {sum([i for i in exp_seq.values()])}")
    return exp_seq, carreer_len


def get_data(CFG, args):
    suffix = f"_{args.mod_type}_it{args.iteration}"
    if args.enforce_monotony != "True":
        suffix += f"_NON_MONOTONIC"
    print("Building datasets...")
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": args.load_dataset,
                 "subsample": -1,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "rep_file": CFG['ppl_rep'],
                 "suffix": suffix,
                 "exp_type": "iter",
                 "is_toy": "False"}
    data_train = StringIndSubDataset(**arguments, split="TRAIN")
    data_valid = StringIndSubDataset(**arguments, split="VALID")
    data_test = StringIndSubDataset(**arguments, split="TEST")
    return data_train, data_valid, data_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod_type", type=str, default="ft") # svm or ft
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--enforce_monotony", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="iter")
    args = parser.parse_args()
    main(args)
