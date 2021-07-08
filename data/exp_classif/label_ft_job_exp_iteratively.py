import ipdb
import argparse
import yaml
import os
import fasttext
import numpy as np
from collections import Counter
from utils.Bunch import Bunch

from tqdm import tqdm
from data.datasets.StringIndSubDataset import StringIndSubDataset
from utils.models import get_metrics, handle_fb_preds, get_metrics_at_k
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
    data_train, data_valid, data_test, train_lookup, valid_lookup, test_lookup = load_datasets(args)
    if args.initial_check == "True" and args.enforce_monotony == "True":
        check_monotonic_dynamic(data_train, train_lookup, "train")
        check_monotonic_dynamic(data_valid, valid_lookup, "valid")
        check_monotonic_dynamic(data_test, test_lookup, "test")

    iteration = args.start_iter
    f1 = 0
    params = (0.1, 5, 1)
    exp_name = get_exp_name(args)

    all_labels = get_all_labels(data_train, data_valid, data_test)
    all_tuples = get_all_tuples(data_train, data_valid, data_test)

    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    all_users = get_all_users(data_train, data_valid, data_test, train_lookup, valid_lookup, test_lookup)

    check_monotonic_dynamic(data_train + data_valid + data_test, all_users, "all")
    print("Features and labels concatenated.")
    changed_this_iter = 1000
    while f1 < args.f1_threshold and iteration < args.max_iter and changed_this_iter > 0:
        train_file, test_file = build_ft_txt_file(args, f'_it{iteration}', data_train, data_valid, data_test)
        print(f"Iteration number: {iteration}")
        print(f"Training classifier on {len(data_train) + len(data_valid)} jobs...")
        classifier = fasttext.train_supervised(input=train_file, lr=params[0], epoch=params[1],
                                               wordNgrams=params[2])
        if iteration == 0:
            init_metrics = test_model_on_all_test_data(classifier, test_file)
            print(f"initial F1 score: {init_metrics[0]['f1_exp']}%")
            class_dist = get_class_dist(all_labels)
            print(f"Initial class dist: {class_dist}")
        iteration += 1
        classifier.save_model(tgt_file)
        print(f"Model saved at {tgt_file}")
        preds, labels = [], []
        changed_this_iter, num_seen = 0, 0
        # FT eval
        # this allows to cover different users from one loop to the next, even if we skip some every loop
        cnt = np.random.randint(args.user_step)
        for user in tqdm(all_users.keys(), desc="parsing users..."):
            if cnt % args.user_step == 0:
                current_user = all_users[user]
                exp_seq_pred = [all_labels[current_user[0]]]
                exp_seq_init = [all_labels[current_user[0]]]
                for job in range(current_user[0] + 1, current_user[1]):
                    num_seen += 1
                    if args.enforce_monotony == "True":
                        prev_exp = all_labels[job - 1]
                        if job == current_user[1] - 1:
                            next_exp = 2  # max possible
                        else:
                            next_exp = all_labels[job + 1]
                        tmp = handle_fb_preds(classifier.predict(all_tuples[job], k=3))
                        pred = int(tmp[0])
                        # relabel current tuple according to evolution constraint
                        exp_seq_init.append(all_labels[job])
                        if prev_exp <= pred <= next_exp:
                            if all_labels[job] != pred:
                                changed_this_iter += 1
                            all_labels[job] = pred
                    else:
                        tmp = handle_fb_preds(classifier.predict(all_tuples[job], k=3))
                        pred = int(tmp[0])
                        if all_labels[job] != pred:
                            changed_this_iter += 1
                        all_labels[job] = pred
                    # keep predictions for evaluation of convergence
                    preds.append(pred)
                    labels.append(all_labels[job])
                    exp_seq_pred.append(all_labels[job])
                if args.enforce_monotony == "True":
                    assert all(exp_seq_pred[i] <= exp_seq_pred[i + 1] for i in range(len(exp_seq_pred) - 1))
                    assert all(exp_seq_init[i] <= exp_seq_init[i + 1] for i in range(len(exp_seq_init) - 1))
            cnt += 1
        print(f"changed this iteration: {changed_this_iter} -- {100 * changed_this_iter / num_seen} % of the jobs seen")
        # metrics = get_metrics(preds, labels, args.exp_levels, f"it_{iteration}")
        metrics = test_model_on_all_test_data(classifier, test_file)
        f1 = metrics[0]["f1_exp"]
        print(f"Iteration: {iteration}, F1 score: {f1}%")
        print(f"initial F1 score: {init_metrics[0]['f1_exp']}%")
        pred_class_dist = get_class_dist(preds)
        label_class_dist = get_class_dist(labels)
        print(f"Class distributions in PREDS: {pred_class_dist}")
        print(f"Class distributions in LABELS: {label_class_dist}")
        word_analysis(args, all_tuples, all_labels, exp_name, iteration)
        save_new_tuples(data_train, data_valid, data_test, all_labels, iteration)

    # the model converged, we test it on the whole dataset
    all_results_test = test_model_on_all_test_data(classifier, test_file)
    all_results_train = test_model_on_all_test_data(classifier, train_file)
    ipdb.set_trace()


def test_model_on_all_test_data(model, test_file):
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
            k = 2
            labels.append(int(tmp.split(" ")[0]))
            pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
            predictions.append(pred)
    preds = np.stack(predictions)
    metrics_at_1 = get_metrics(preds[:, 0], labels, len(model.labels), "exp")
    metrics_at_k = get_metrics_at_k(preds, labels, len(model.labels), "exp @2")
    return metrics_at_1, metrics_at_k


def get_all_labels(data_train, data_valid, data_test):
    all_labels = []
    for i in tqdm(data_train.tuples, desc="parsing train tuples ot gather labels..."):
        all_labels.append(i["exp_index"])
    for i in tqdm(data_valid.tuples, desc="parsing valid tuples ot gather labels..."):
        all_labels.append(i["exp_index"])
    for i in tqdm(data_test.tuples, desc="parsing test tuples ot gather labels..."):
        all_labels.append(i["exp_index"])
    return all_labels


def get_all_tuples(data_train, data_valid, data_test):
    all_tuples = []
    for i in tqdm(data_train.tuples, desc="parsing train tuples ot gather tupless..."):
        all_tuples.append(i["words"])
    for i in tqdm(data_valid.tuples, desc="parsing valid tuples ot gather tupless..."):
        all_tuples.append(i["words"])
    for i in tqdm(data_test.tuples, desc="parsing test tuples ot gather tupless..."):
        all_tuples.append(i["words"])
    return all_tuples


def build_ft_txt_file(args, suffix, dataset_train, dataset_valid, dataset_test):
    if args.enforce_monotony != "True":
        suffix += f"_NON_MONOTONIC"
    tgt_file = os.path.join(CFG["gpudatadir"],
                            f"ft_classif_supervised_ind20_exp{args.exp_levels}_{args.exp_type}{suffix}.test")
    if os.path.isfile(tgt_file):
        os.system('rm ' + tgt_file)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file, dataset_test.tuples, f"exp", "test")

    tgt_file_exp_model = os.path.join(CFG["gpudatadir"],
                                      f"ft_classif_supervised_ind20_exp{args.exp_levels}_{args.exp_type}{suffix}.train")
    print(tgt_file_exp_model)
    if os.path.isfile(tgt_file_exp_model):
        os.system('rm ' + tgt_file_exp_model)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file_exp_model, dataset_train,
                             f"exp", "train")
    write_in_file_with_label(args, tgt_file_exp_model, dataset_valid,
                             f"exp", "valid")

    return tgt_file_exp_model, tgt_file


def write_in_file_with_label(args, tgt_file, dataset, att_type, split):
    with open(tgt_file, 'a+') as f:
        tmp = []
        for item in tqdm(dataset, desc="Parsing train " + split + " dataset for " + att_type + "..."):
            if split == "test":
                job_str = item['words']
                att = item['exp_index']
            else:
                job_str = item[0]
                att = item[-1]
            final_str = f"__label__{att} {job_str} \n"
            f.write(final_str)
            tmp.append(att)
        print(len(set(tmp)))


def check_monotonic_dynamic(data, lookup, split):
    if split == "all":
        all_labels = [i[-1] for i in data]
    else:
        all_labels = [i["exp_index"] for i in data.tuples]
    for cnt, user in enumerate(tqdm(lookup.keys(), desc=f"initial monotonicity check on split {split} users...")):
        current_user = lookup[user]
        exp_seq_init = [all_labels[current_user[0]]]
        for job in range(current_user[0] + 1, current_user[1]):
            exp_seq_init.append(all_labels[job])
        assert all(exp_seq_init[i] <= exp_seq_init[i + 1] for i in range(len(exp_seq_init) - 1))


def word_analysis(args, all_tuples, all_labels, exp_name, iteration):
    class_txt = get_jobs_str_per_class(args, all_tuples, all_labels)
    word_counter = {k: Counter() for k in range(args.exp_levels)}
    for k in tqdm(word_counter.keys(), desc="counting words per class"):
        for word in class_txt[k].split(" "):
            word_counter[k][word] += 1
    mc_words_per_class = {}
    # min_len = min([len(word_counter[k] for k in range(args.exp_levels))])
    for k in word_counter.keys():
        mc_words_per_class[k] = set([i[0] for i in word_counter[k].most_common(500)])
    words_unique_to_class0 = mc_words_per_class[0] - mc_words_per_class[1] - mc_words_per_class[2]
    words_unique_to_class1 = mc_words_per_class[1] - mc_words_per_class[0] - mc_words_per_class[2]
    words_unique_to_class2 = mc_words_per_class[2] - mc_words_per_class[1] - mc_words_per_class[0]
    tmp = [words_unique_to_class0, words_unique_to_class1, words_unique_to_class2]
    for k, word_list in zip(mc_words_per_class.keys(), tmp):
        get_word_clouds_for_class(word_counter[k], word_list, k, exp_name + f"_ft_it{iteration}")
    print("word clouds saved!")


def get_word_clouds_for_class(class_text, words_to_print, class_num, exp_name):
    string = ""
    for word, count in class_text.items():
        if word in words_to_print:
            for i in range(count):
                string += f" {word}"
    print("Samples sorted by label...")
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", collocations=False).generate(
        string)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"img/WORDCLOUD_{exp_name}_class{class_num}.png")
    wordcloud.to_file(f"img/WORDCLOUD_{exp_name}_class{class_num}.png")
    plt.close()


def get_jobs_str_per_class(args, all_tuples, all_labels):
    # sort job id per class
    class_index = {k: [] for k in range(args.exp_levels)}
    for num, _ in enumerate(tqdm(all_labels)):
        class_index[all_labels[num]].append(num)
    rev_class_index = {}
    for k, v in tqdm(class_index.items(), desc="Building reversed index for topics..."):
        for job_index in v:
            rev_class_index[job_index] = k
    class_txt = {k: "" for k in range(args.exp_levels)}
    indices_to_get = range(0, len(all_tuples), 100)
    for num, ind in enumerate(tqdm(indices_to_get, desc="sorting jobs by class on all samples")):
        class_txt[rev_class_index[num]] += f" {all_tuples[ind]}"
    return class_txt


def save_new_tuples(data_train, data_valid, data_test, all_labels, iteration):
    tuples_train = []
    labels_train = all_labels[:len(data_train)]
    for num, label in enumerate(tqdm(labels_train, desc="relabel train tuples...")):
        new = {"ind_index": data_train.tuples[num]["ind_index"],
               "exp_index": label,
               "words": data_train.tuples[num]["words"]}
        tuples_train.append(new)

    tuples_valid = []
    labels_valid = all_labels[len(data_train):len(data_train) + len(data_valid)]
    for num, label in enumerate(tqdm(labels_valid, desc="relabel valid tuples...")):
        new = {"ind_index": data_valid.tuples[num]["ind_index"],
               "exp_index": label,
               "words": data_valid.tuples[num]["words"]}
        tuples_valid.append(new)

    labels_test = all_labels[len(data_train) + len(data_valid):-1]
    tuples_test = []
    for num, label in enumerate(tqdm(labels_test, desc="relabel test tuples...")):
        new = {"ind_index": data_test.tuples[num]["ind_index"],
               "exp_index": label,
               "words": data_test.tuples[num]["words"]}
        tuples_test.append(new)
    save_new_tuples_per_split(data_train, "TRAIN", iteration)
    save_new_tuples_per_split(data_valid, "VALID", iteration)
    save_new_tuples_per_split(tuples_test, "TEST", iteration)


def save_new_tuples_per_split(tuple_list, split, iteration):
    suffix = f"_ft_it{iteration}"
    if args.enforce_monotony != "True":
        suffix += f"_NON_MONOTONIC"
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "exp_type": "iter",
                 "rep_file": CFG['ppl_rep'],
                 "suffix": f"_ft_it{iteration-1}",
                 "split": split,
                 "is_toy": "False"}
    tmp = StringIndSubDataset(**arguments)
    tmp.name = f"StringIndSubDataset_{args.exp_levels}exp_iter_maxlen{args.max_len}_{split}"
    tmp.save_new_tuples(tuple_list, suffix)


def get_class_dist(class_list):
    cnt = Counter()
    total = len(class_list)
    for i in class_list:
        cnt[i] += 100 / total
    return cnt


def load_datasets(args):
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    suffix = f"_ft_it{args.start_iter}"
    if args.user_step != 1:
        suffix += f"_step{args.user_step}"
    if args.enforce_monotony != "True":
        suffix += f"_NON_MONOTONIC"
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": args.load_dataset,
                 "subsample": -1,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "rep_file": CFG['ppl_rep'],
                 "suffix": suffix,
                 "exp_type": "iter",
                 "is_toy": "False"}
    for split in splits:
        datasets.append(StringIndSubDataset(**arguments, split=split))
    train_lookup = datasets[0].user_lookup
    valid_lookup = datasets[1].user_lookup
    print(f"train and valid lookup len : {len(train_lookup) + len(valid_lookup)}")
    test_lookup = datasets[-1].user_lookup
    return datasets[0], datasets[1], datasets[2], train_lookup, valid_lookup, test_lookup


def get_exp_name(args):
    exp_name = f"FT_label_iter_{args.exp_levels}exp_{args.exp_type}"
    if args.user_step != 1:
        exp_name += f"_step{args.user_step}"
    if args.enforce_monotony != "True":
        exp_name += f"_NON_MONOTONIC"
    return exp_name


def get_all_users(data_train, data_valid, data_test, train_lu, valid_lu, test_lu):
    offset = len(data_train)
    offset_valid_lookup = {}
    for k, v in valid_lu.items():
        offset_valid_lookup[k] = [v[0] + offset, v[1] + offset]
    assert v[1] + offset <= len(data_train) + len(data_valid)
    offset = len(data_train) + len(data_valid)
    offset_test_lookup = {}
    for k, v in test_lu.items():
        offset_test_lookup[k] = [v[0] + offset, v[1] + offset]
    assert v[1] + offset <= len(data_train) + len(data_valid) + len(data_test)
    return {**train_lu, **offset_valid_lookup, **offset_test_lookup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample_users", type=int, default=-1)
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--subsample_jobs", type=int, default=-1)
    # parser.add_argument("--train_user_len", type=int, default=256160)# max: 256160
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--user_step", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--f1_threshold", type=int, default=80)
    parser.add_argument("--exp_type", type=str, default="iter")
    parser.add_argument("--enforce_monotony", type=str, default="True")
    parser.add_argument("--ind_sub", type=str, default="True")
    parser.add_argument("--initial_check", type=str, default="False")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
