import ipdb
import argparse
import yaml
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
    json_file = CFG["ppl_rep"]
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "False",
                 "subsample": -1,
                 "max_len": 32,
                 "exp_levels": args.exp_levels,
                 "rep_file": json_file,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    for split in splits:
        StringIndSubDataset(**arguments, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
