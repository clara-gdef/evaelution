import os
import pickle as pkl
from collections import Counter
import ipdb

import yaml
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score

from data.datasets.StringIndSubDataset import StringIndSubDataset


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    file_root = f"bow_jobs_pre_proced_{args.exp_type}_{args.exp_levels}exp_maxlen{args.max_len}"
    if args.add_ind_name == "True":
        file_root += "_indName"
    suffix = args.dataset_suffix
    if args.load_data == "True":
        print("Loading data...")
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TRAIN.pkl"), 'rb') as f_name:
            data_train = pkl.load(f_name)
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TEST.pkl"), 'rb') as f_name:
            data_test = pkl.load(f_name)
        with open(os.path.join(CFG["gpudatadir"],
                               f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_{suffix}.pkl"),
                  'rb') as f_name:
            class_weights = pkl.load(f_name)
    else:
        print("Building datasets...")
        ppl_file = CFG["ppl_rep"]
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": "True",
                     "subsample": args.subsample,
                     "max_len": args.max_len,
                     "exp_levels": args.exp_levels,
                     "rep_file": ppl_file,
                     "suffix": suffix,
                     "exp_type": args.exp_type,
                     "is_toy": args.toy_dataset}
        dataset_train = StringIndSubDataset(**arguments, split="TRAIN")
        dataset_valid = StringIndSubDataset(**arguments, split="VALID")
        dataset_test = StringIndSubDataset(**arguments, split="TEST")
        dataset_train.tuples.extend(dataset_valid.tuples)
        class_weights = get_class_weights(dataset_train)
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words("french"))
        stop_words.add("les")

        np.random.shuffle(dataset_train.tuples)

        if args.add_ind_name == "True":
            jobs, labels_exp, labels_ind = pre_proc_data_ind(dataset_test.ind_dict, dataset_train, tokenizer, stop_words)
            data_train = {"jobs": jobs, "labels_exp": labels_exp, "labels_ind": labels_ind}
            jobs, labels_exp, labels_ind = pre_proc_data_ind(dataset_test.ind_dict, dataset_test, tokenizer, stop_words)
            data_test = {"jobs": jobs, "labels_exp": labels_exp, "labels_ind": labels_ind}

        else:
            jobs, labels_exp, labels_ind = pre_proc_data(dataset_train, tokenizer, stop_words)
            data_train = {"jobs": jobs, "labels_exp": labels_exp, "labels_ind": labels_ind}
            jobs, labels_exp, labels_ind = pre_proc_data(dataset_test, tokenizer, stop_words)
            data_test = {"jobs": jobs, "labels_exp": labels_exp, "labels_ind": labels_ind}

        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TRAIN.pkl"), 'wb') as f_name:
            pkl.dump(data_train, f_name)
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TEST.pkl"), 'wb') as f_name:
            pkl.dump(data_test, f_name)
        with open(os.path.join(CFG["gpudatadir"],
                               f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_{suffix}.pkl"),
                  'wb') as f_name:
            pkl.dump(class_weights, f_name)
    return data_train, data_test, class_weights


def pre_proc_data(data, tokenizer, stop_words):
    labels_exp, labels_ind, jobs = [], [], []
    for job in tqdm(data, desc="Parsing profiles..."):
        labels_exp.append(int(job[2]))
        labels_ind.append(int(job[1]))
        cleaned_ab = [w.lower() for w in tokenizer.tokenize(job[0]) if (w not in stop_words) and (w != "")]
        jobs.append(" ".join(cleaned_ab))
    return jobs, labels_exp, labels_ind


def pre_proc_data_ind(ind_dict, data, tokenizer, stop_words):
    labels_exp, labels_ind, jobs = [], [], []
    for job in tqdm(data, desc="Parsing profiles..."):
        labels_exp.append(int(job[2]))
        labels_ind.append(int(job[1]))
        cleaned_ab = [w.lower() for w in tokenizer.tokenize(job[0]) if (w not in stop_words) and (w != "")]
        tmp = ["_".join(ind_dict[job[1]].split(" "))]
        tmp.extend(cleaned_ab)
        jobs.append(" ".join(tmp))
    return jobs, labels_exp, labels_ind


def get_class_weights(dataset_train):
    exp, ind = Counter(), Counter()
    for job in tqdm(dataset_train, desc="getting class weights..."):
        exp[job[2]] += 1
        ind[job[1]] += 1
    exp_weights, ind_weights = dict(), dict()
    for k in exp.keys():
        exp_weights[k] = exp[k] / len(dataset_train)
    for k in ind.keys():
        ind_weights[k] = ind[k] / len(dataset_train)
    return {"exp": exp_weights, "ind": ind_weights}


def fit_vectorizer(args, input_data):
    mf = args.max_features if args.max_features != 0 else None
    if args.tfidf == "True":
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=args.max_df, min_df=args.min_df, max_features=mf)
    else:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=args.max_df, min_df=args.min_df, max_features=mf)
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform([np.str_(x) for x in input_data])
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features, vectorizer


def train_svm(data, labels, class_weights, kernel):
    model = SVC(kernel=kernel, class_weight=class_weights, verbose=True, max_iter=100)
    print("Fitting SVM...")
    model.fit(data, labels)
    print("SVM fitted!")
    return model


def train_nb(data, labels, class_weights):
    priors = [i[1] for i in sorted(class_weights.items())]
    model = MultinomialNB(class_prior=priors)
    print("Fitting Naive Bayes...")
    model.fit(data, labels)
    print("Naive Bayes fitted!")
    return model


def test_for_att(args, class_dict, att_type, labels, model, features, split):
    num_c = len(class_dict[att_type])
    handle = f"{att_type} {split} {args.model}"
    preds, preds_at_k, k = get_predictions(args, model, features, labels, att_type)
    res_at_1 = eval_model(labels, preds, num_c, handle)
    res_at_k = eval_model(labels, preds_at_k, num_c, f"{handle}_@{k}")
    return {**res_at_1, **res_at_k}


def get_predictions(args, model, features, labels, att_type):
    if args.model == "svm":
        predictions = model.decision_function(features)
    elif args.model == "nb":
        predictions = model.predict_log_proba(features)
    if att_type == "exp":
        k = 2
    else:
        k = 5
    predictions_at_1 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_1.append(get_pred_at_k(sample, lab, 1))
    predictions_at_10 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_10.append(get_pred_at_k(sample, lab, k))
    return predictions_at_1, predictions_at_10, k


def eval_model(labels, preds, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_pred_at_k(pred, label, k):
    ranked = np.argsort(pred, axis=-1)
    largest_indices = ranked[::-1][:len(pred)]
    new_pred = largest_indices[0]
    if label in largest_indices[:k]:
        new_pred = label
    return new_pred
