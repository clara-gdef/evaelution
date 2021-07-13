import argparse
import os
import pickle as pkl
import time
from collections import Counter

import ipdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from wordcloud import WordCloud
from data.exp_classif.label_job_exp_iteratively import get_subset_data_and_labels

from data.datasets.StringIndSubDataset import StringIndSubDataset
from utils.bow import train_nb, test_for_att
from utils.models import get_metrics
from sklearn.naive_bayes import MultinomialNB


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
    data_train, data_valid, data_test, train_lookup, valid_lookup, test_lookup = load_datasets(args, args.start_iter)
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)

    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words("french"))
    stop_words.add("les")

    cleaned_profiles_train, labels_exp_train, _ = pre_proc_data(data_train, tokenizer, stop_words)
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=1.0, min_df=1, max_features=12000)
    print("Fitting vectorizer...")
    train_features = vectorizer.fit_transform(cleaned_profiles_train)
    print("Vectorizer Fitted.")

    if args.start_iter == 0:
        subset_train_data, subset_train_labels, user_trains = get_subset_data_and_labels(train_features.toarray(),
                                                                                         labels_exp_train,
                                                                                         train_lookup,
                                                                                         args.train_user_len)
        class_weights = get_class_dist(data_train)
        priors = [i[1] for i in sorted(class_weights.items())]
        classifier = MultinomialNB(class_prior=priors)
        print("Fitting Naive Bayes...")
        classifier.fit(subset_train_data, subset_train_labels)
        print("Naive Bayes fitted!")
    else:
        classifier = joblib.load(f"{tgt_file}_exp_nb_it{args.start_iter}.joblib")

    all_results = test_model_on_all_test_data(args, classifier, vectorizer, tokenizer, stop_words, args.start_iter)
    ipdb.set_trace()


def test_model_on_all_test_data(args, model, vectorizer, tokenizer, stop_words, iteration):
    data_train, data_valid, data_test, train_lookup, valid_lookup, test_lookup = load_datasets(args, iteration)
    cleaned_profiles_test, labels_exp_test, _ = pre_proc_data(data_test, tokenizer, stop_words)
    test_features = vectorizer.transform(cleaned_profiles_test)
    preds = model.predict(test_features.toarray())
    assert len(preds) == len(labels_exp_test)
    metrics = get_metrics(preds, labels_exp_test, args.exp_levels, f"whole")
    print(metrics)
    ipdb.set_trace()
    return metrics


def load_datasets(args, iteration):
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    suffix = f"_nb_it{iteration}"
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
    test_lookup = datasets[-1].user_lookup
    return datasets[0], datasets[1], datasets[2], train_lookup, valid_lookup, test_lookup


def get_exp_name(args):
    exp_name = f"label_iter_loose_{args.exp_levels}exp_{args.exp_type}_train{args.train_user_len}"
    if args.subsample_users != -1:
        exp_name += f"_eval{args.subsample_users}"
    if args.user_step != 1:
        exp_name += f"_step{args.user_step}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


def pre_proc_data(data, tokenizer, stop_words):
    labels_exp, labels_ind, jobs = [], [], []
    for job in tqdm(data, desc="Parsing profiles..."):
        labels_exp.append(int(job[2]))
        labels_ind.append(int(job[1]))
        cleaned_ab = [w.lower() for w in tokenizer.tokenize(job[0]) if (w not in stop_words) and (w != "")]
        jobs.append(" ".join(cleaned_ab))
    return jobs, labels_exp, labels_ind


def get_class_dist(class_list):
    cnt = Counter()
    total = len(class_list)
    for i in class_list:
        cnt[i[-1]] += 100 / total
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample_users", type=int, default=-1)
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--train_user_len", type=int, default=2000)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--user_step", type=int, default=10)
    parser.add_argument("--start_iter", type=int, default=100)
    parser.add_argument("--f1_threshold", type=int, default=80)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--ind_sub", type=str, default="True")
    parser.add_argument("--initial_check", type=str, default="False")
    parser.add_argument("--tfidf", type=str, default="True")
    parser.add_argument("--model", type=str, default="nb")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
