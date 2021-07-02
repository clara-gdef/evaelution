import argparse
import yaml
import os
import joblib
import ipdb
import numpy
import pickle as pkl
from tqdm import tqdm
from sklearn.preprocessing import scale
from utils.bow import fit_vectorizer, train_svm, train_nb, get_labelled_data, test_for_att


def init(args):
    with ipdb.launch_ipdb_on_exception():
        print('Commencing data gathering...')
        data_train, data_test, class_weights = get_labelled_data(args)
        print('Data loaded')
        main(args, data_train, data_test, class_weights)


def main(args, data_train, data_test, class_dict):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    # TRAIN
    if args.subsample > 0:
        data_train, data_test = subsample_according_to_class_weight(args, data_train, data_test, class_dict, args.subsample)
    train_features, vectorizer = fit_vectorizer(args, data_train["jobs"])
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    if args.load_model == "True":
        model = joblib.load(f"{tgt_file}_{args.att_type}.joblib")
        vectorizer = joblib.load(f"{tgt_file}_vectorizer_{args.model}_{args.kernel}.joblib")
    else:
        if args.model == "svm":
            model = train_svm(scale(train_features[:args.subsample]), data_train[f"labels_{args.att_type}"][:args.subsample], class_dict[args.att_type], args.kernel)
        elif args.model == "nb":
            model = train_nb(scale(train_features[:args.subsample]), data_train[f"labels_{args.att_type}"][:args.subsample], class_dict[args.att_type])
        else:
            raise Exception("Wrong model type specified, can be either svm or nb")
        if args.save_model == "True":
            joblib.dump(model, f"{tgt_file}_{args.att_type}.joblib")
            joblib.dump(vectorizer, f"{tgt_file}_vectorizer_{args.model}_{args.kernel}.joblib")
    # TEST
    test_features = vectorizer.transform(data_test["jobs"])

    res_test = test_for_att(args, class_dict, args.att_type, data_test[f"labels_{args.att_type}"][:args.subsample], model, test_features[:args.subsample].todense(), "TEST")
    res_train = test_for_att(args, class_dict, args.att_type, data_train[f"labels_{args.att_type}"][:args.subsample], model, train_features[:args.subsample], "TRAIN")

    all_res = {**res_test, **res_train}
    print(f"EXP NAME: {exp_name}")
    print(f"NUM FEATURES: {model.coef_.shape[1]}")
    print(select_relevant_keys(f"{args.att_type} TEST {args.model}", all_res))
    #ipdb.set_trace()
    return all_res


def select_relevant_keys(handle, test_results):
    relevant_results = {}
    for k in test_results.keys():
        if k.startswith(f"acc_{handle}") or k.startswith(f"f1_{handle}"):
            relevant_results[k] = test_results[k]
    return relevant_results


def subsample_according_to_class_weight(args, data_train, data_test, class_dict, subsample):
    tgt_file = os.path.join(CFG["gpudatadir"], f"subsampled_{subsample}_bow_jobs_pre_proced_{args.exp_type}_{args.exp_levels}exp.pkl")
    if os.path.isfile(tgt_file):
        with open(tgt_file, "rb") as f:
            dat = pkl.load(f)
        sub_train = dat["train"]
        sub_test = dat["test"]
    else:
        ind_weights = class_dict["ind"]
        num_sample_per_class = {k: int(v*subsample) for k, v in ind_weights.items()}

        sub_train = {"jobs": [], "labels_exp": [], "labels_ind": []}
        sampled_index = []
        for k in tqdm(num_sample_per_class.keys(), desc="Subsampling train data according to industry class ratio"):
            class_counter = 0
            while class_counter < num_sample_per_class[k]:
                rdm_index = numpy.random.randint(len(data_train["jobs"]))
                if (data_train["labels_ind"][rdm_index] == k) and (rdm_index not in sampled_index):
                    sub_train["jobs"].append(data_train["jobs"][rdm_index])
                    sub_train["labels_ind"].append(data_train["labels_ind"][rdm_index])
                    sub_train["labels_exp"].append(data_train["labels_exp"][rdm_index])
                    class_counter += 1
                    sampled_index.append(rdm_index)

        num_sample_per_class = {k: int(v*subsample/2) for k, v in ind_weights.items()}
        sub_test = {"jobs": [], "labels_exp": [], "labels_ind": []}
        sampled_index = []
        ipdb.set_trace()
        for k in tqdm(num_sample_per_class.keys(), desc="Subsampling test data according to industry class ratio"):
            class_counter = 0
            while class_counter < num_sample_per_class[k]:
                rdm_index = numpy.random.randint(len(data_test["jobs"]))
                if (data_test["labels_ind"][rdm_index] == k) and (rdm_index not in sampled_index):
                    sub_test["jobs"].append(data_test["jobs"][rdm_index])
                    sub_test["labels_ind"].append(data_test["labels_ind"][rdm_index])
                    sub_test["labels_exp"].append(data_test["labels_exp"][rdm_index])
                    class_counter += 1
                    sampled_index.append(rdm_index)
        tmp = {"train": sub_train, "test": sub_test}
        with open(tgt_file, "wb") as f:
            pkl.dump(tmp, f)
    return sub_train, sub_test


def get_exp_name(args):
    exp_name = f"{args.model}_{args.exp_levels}exp_{args.exp_type}_maxlen{args.max_len}"
    return exp_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type=str, default="True")
    parser.add_argument("--min_df", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="svm") # svm or nb
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--max_features", type=int, default=3000)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--load_model", type=str, default="False")
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--dataset_suffix", type=str, default="")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--att_type", type=str, default="exp") # ind or exp
    parser.add_argument("--exp_type", type=str, default="uniform") # uniform
    parser.add_argument("--exp_levels", type=int, default=3) # 5 of 3
    parser.add_argument("--kernel", type=str, default="linear")
    args = parser.parse_args()
    init(args)
