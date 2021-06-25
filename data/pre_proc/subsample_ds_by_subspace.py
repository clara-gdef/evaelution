import ipdb
import argparse
import yaml
import os
from tqdm import tqdm
import pickle as pkl
from data.datasets import StringIndSubDataset


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
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "rep_file": None,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    ds_train = StringIndSubDataset(**arguments, split="TRAIN")
    ds_test = StringIndSubDataset(**arguments, split="TEST")
    industries = ds_train.ind_dict
    all_groups = get_all_groups(industries, args.exp_levels)
    sub_train = get_subgroup_for_split(all_groups, ds_train, "train")
    sub_test = get_subgroup_for_split(all_groups, ds_test, "test")
    ipdb.set_trace()


def get_all_groups(inds, exp_levels):
    all_groups = []
    for ind_index, ind_name in inds.items():
        for i in range(exp_levels):
            tmp = (ind_index, i)
            all_groups.append(tmp)
    return all_groups


def get_subgroup_for_split(all_groups, ds, split):
    sub = []
    for couple in tqdm(all_groups, desc=f"building visualisation subgroup in {split}set"):
        cnt = 0
        for job in ds.tuples:
            if job["ind_index"] == couple[0] and job["exp_index"] == couple[1] and cnt < args.num_point_per_group:
                sub.append(job)
                cnt += 1
    tgt_file = os.path.join(CFG["gpudatadir"], f"viz_subgroup_{split}.pkl")
    with open(tgt_file, 'wb') as f:
        pkl.dump(sub, f)
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--num_point_per_group", type=int, default=5)
    args = parser.parse_args()
    init(args)
