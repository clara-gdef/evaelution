import argparse
import fasttext
import ipdb
from tqdm import tqdm
import yaml
import pickle as pkl
import os
import numpy as np
from data.datasets.StringIndSubDataset import StringIndSubDataset
from utils.models import handle_fb_preds, get_metrics, get_metrics_at_k


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        params, suffix, train_file, test_file = get_param_and_suffix(args)
        tgt_file = os.path.join(CFG["modeldir"], f"ft_att_classifier_{suffix}.bin")
        if args.TRAIN == "True":
            model = fasttext.train_supervised(input=train_file, lr=params[0], epoch=params[1], wordNgrams=params[2])
            model.save_model(tgt_file)
            print(f"Model saved at {tgt_file}")
        if args.TEST == "True":
            label_dict = load_label_dict(args)
            if args.TRAIN == "False":
                model = fasttext.load_model(tgt_file)
            metrics_at_1, metrics_at_2 = eval_procedure(test_file, model, label_dict)
            print("TEST METRICS")
            print(metrics_at_1)
            print(metrics_at_2)
            print("TRAIN METRICS")
            metrics_at_1, metrics_at_2 = eval_procedure(train_file, model, label_dict)
            print(metrics_at_1)
            print(metrics_at_2)


def load_label_dict(args):
    if args.att_type == "ind":
        with open(os.path.join(CFG["gpudatadir"], "20_industry_dict.pkl"), 'rb') as f_name:
            industry_dict = pkl.load(f_name)
        rev_ind_dict = dict()
        for k, v in industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        return rev_ind_dict
    else:
        with open(os.path.join(CFG["gpudatadir"], "exp_dict_fr_3.pkl"), 'rb') as f_name:
            exp_dict = pkl.load(f_name)
    rev_exp_dict = {v: k for k, v in exp_dict.items()}
    return rev_exp_dict


def eval_procedure(test_file, model, label_dict):
    print("Testing...")
    num_lines = 0
    with open(test_file, 'r') as test_f:
        for line in test_f:
            num_lines += 1
    labels = []
    predictions = []
    with open(test_file, 'r') as test_f:
        pbar = tqdm(test_f, total=num_lines)
        for line in pbar:
            tmp = line.split("__label__")[1]
            if args.att_type == "ind":
                k = 5
                labels.append(label_dict[tmp.split(" ")[0]])
                pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
                predictions.append([label_dict[p] for p in pred])
            else:
                k = 2
                labels.append(int(tmp.split(" ")[0]))
                pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
                predictions.append(pred)
    preds = np.stack(predictions)
    metrics_at_1 = get_metrics(preds[:, 0], labels, len(model.labels), f"{args.att_type}")
    metrics_at_k = get_metrics_at_k(preds, labels, len(model.labels), f"{args.att_type} @{k}")
    ipdb.set_trace()
    return metrics_at_1, metrics_at_k


def get_param_and_suffix(args):
    suffix = f"{args.att_type}"
    if args.att_type == "exp":
        suffix += f"{args.exp_levels}_{args.exp_type}"
    if args.add_ind_name == "True":
        suffix += "_IndName"
    suffix += args.dataset_suffix

    params = (0.1, 5, 1)

    train_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{suffix}.train")
    test_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{suffix}.test")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        print("files exist, ready to be loaded.")
    else:
        print("Txt files for training do not exist, building them...")
        build_txt_files(args, suffix)
    return params, suffix, train_file, test_file


def build_txt_files(args, suffix):
    ppl_file = CFG["ppl_rep"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": args.subsample,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "rep_file": ppl_file,
                 "suffix": args.dataset_suffix,
                 "exp_type": args.exp_type,
                 "is_toy": args.toy_dataset}
    train_dataset = StringIndSubDataset(**arguments, split="TRAIN")
    valid_dataset = StringIndSubDataset(**arguments, split="VALID")
    test_dataset = StringIndSubDataset(**arguments, split="TEST")

    tgt_file_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{suffix}.train")
    print(tgt_file_model)
    build_train_file(args, tgt_file_model, train_dataset, valid_dataset, args.att_type)

    tgt_file_test_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{suffix}.test")
    if os.path.isfile(tgt_file_test_model):
        os.system('rm ' + tgt_file_test_model)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file_test_model, test_dataset, args.att_type, "test", train_dataset.ind_dict)
    print("File " + tgt_file_test_model + " built.")


def build_train_file(args, tgt_file, train_dataset, valid_dataset, att_type):
    ind_dict = train_dataset.ind_dict
    if os.path.isfile(tgt_file):
        os.system('rm ' + tgt_file)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file, train_dataset, att_type, "train", ind_dict)
    write_in_file_with_label(args, tgt_file, valid_dataset, att_type, "valid", ind_dict)
    print("File " + tgt_file + " built.")


def write_in_file_with_label(args, tgt_file, dataset, att_type, split, ind_dict):
    with open(tgt_file, 'a+') as f:
        tmp = []
        for item in tqdm(dataset, desc="Parsing train " + split + " dataset for " + att_type + "..."):
            job_str = item[0]
            if args.add_ind_name == "True":
                tmp2 = job_str
                job_str = "_".join(ind_dict[item[1]].split(" "))
                job_str += " " + tmp2
            att = get_attribute(att_type, dataset, item)
            final_str = f"__label__{att} {job_str} \n"
            f.write(final_str)
            tmp.append(att)
        print(len(set(tmp)))


def get_attribute(att_type, dataset, item):
    if att_type == "ind":
        industry_tmp = dataset.ind_dict[item[-2]]
        if len(industry_tmp.split(" ")) > 1:
            att = "_".join(industry_tmp.split(" "))
        else:
            att = industry_tmp
    elif att_type == "exp":
        att = int(item[-1])
    else:
        raise Exception(att_type)
    return att


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--exp_type", type=str, default='uniform')
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--add_ind_name", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--dataset_suffix", type=str, default="")# can be ind or exp
    parser.add_argument("--att_type", type=str, default="exp")# can be ind or exp
    args = parser.parse_args()
    main(args)
