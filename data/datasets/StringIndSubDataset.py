import os
import pickle as pkl

import ipdb
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class StringIndSubDataset(Dataset):
    def __init__(self, data_dir, load, subsample, max_len, is_toy, exp_levels, exp_type, rep_file, split, suffix=""):
        self.max_len = max_len
        self.is_toy = is_toy
        self.subsample = subsample
        self.split = split
        self.datadir = data_dir
        self.exp_levels = exp_levels
        self.exp_type = exp_type
        self.name = f"StringIndSubDataset_{self.exp_levels}exp_{self.exp_type}_no_unk_{split}{suffix}"
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(subsample, split, "")
        else:
            print("Loading data files...")
            with open(os.path.join(self.datadir, f"exp_dict_fr_{self.exp_levels}.pkl"), 'rb') as f_name:
                self.exp_dict = pkl.load(f_name)
            ind_file = "20_industry_dict.pkl"
            with open(os.path.join(data_dir, ind_file), 'rb') as f_name:
                industry_dict = pkl.load(f_name)
            prev_ind_file = "ind_class_dict.pkl"
            with open(os.path.join(data_dir, prev_ind_file), 'rb') as f_name:
                self.prev_ind_dict = pkl.load(f_name)
            with open(os.path.join(self.datadir, "ind_map_to_subsampled.pkl"), 'rb') as f:
                self.ind_map_to_subsampled = pkl.load(f)
            print("Data files loaded.")
            self.rev_ind_map_to_subsampled = {v: k for k, v in self.ind_map_to_subsampled.items()}
            self.ind_dict = industry_dict
            self.rev_ind_dict = {v: k for k, v in industry_dict.items()}
            rep_file += f"_{split.upper()}_100.pkl"
            self.tuples, self.user_lookup = self.build_ppl_tuples(rep_file, split)
            if self.exp_type == "uniform" or self.exp_type == "iter":
                self.check_monotonicity()
            self.save_dataset("", subsample)

        if subsample != -1 and self.is_toy == "False":
            np.random.shuffle(self.tuples)
            tmp = self.tuples[:subsample]
            self.tuples = tmp

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["words"], \
            self.tuples[idx]["ind_index"], \
            self.tuples[idx]["exp_index"]

    def save_dataset(self, suffix, subsample):
        if self.is_toy == "True":
            print("Subsampling dataset...")
            new_tuples = []
            np.random.shuffle(self.tuples)
            retained_tuples = self.tuples[:subsample]
            for i in range(100):
                if len(retained_tuples) == 1:
                    new_tuples.append(retained_tuples[0])
                else:
                    new_tuples.extend(retained_tuples)
            self.tuples = new_tuples
            print(f"len tuples in save_dataset: {len(self.tuples)}")
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = self.get_tgt_file(suffix, subsample)

        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)
        print(f"Dataset saved at {tgt_file}")

    def load_dataset(self, subsample, split, suffix):
        tgt_file = self.get_tgt_file(suffix, subsample)
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)

        for key in tqdm(ds_dict, desc="Loading attributes from save..."):
            vars(self)[key] = ds_dict[key]
        print("Dataset load from : " + tgt_file)

    def build_ppl_tuples(self, rep_file, split):
        user_lookup = {}
        tuples = []
        counter = 0
        with open(os.path.join(self.datadir, rep_file), 'rb') as f:
            data = pkl.load(f)
        for person in tqdm(data, desc=f"Building tuple list for {self.split} split..."):
            id_p = person[0]
            ind = self.handle_ind(person[1])
            if person[1] != "" and ind in self.ind_dict.keys():
                sorted_jobs = sorted(person[-1], key=lambda k: k['from'])
                if self.exp_type == "uniform":
                    exp = self.get_uniform_experience(len(sorted_jobs))
                elif self.exp_type == "iter":
                    ipdb.set_trace()
                else:
                    raise Exception("exp_type provided not supported. Can only support uniform or iteratively labelled exp atm.")
                for num, job in enumerate(sorted_jobs):
                    new_job = dict()
                    new_job["ind_index"] = ind
                    if self.exp_type == "uniform":
                        new_job["exp_index"] = exp[num]
                    elif self.exp_type == "iter":
                        ipdb.set_trace()
                    else:
                        raise Exception("exp_type provided not supported. Can only support uniform or iteratively labelled exp atm.")
                    new_job["words"] = self.tokenize_job(job["job"])
                    tuples.append(new_job)
                user_lookup[id_p] = [counter, counter + len(sorted_jobs) - 1]
                counter += len(sorted_jobs)
        return tuples, user_lookup

    def get_tgt_file(self, suffix, subsample):
        if self.is_toy != "True":
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}.pkl")
        else:
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}_{subsample}.pkl")
        return tgt_file

    def handle_ind(self, ind_name):
        if ind_name == "":
            return -1
        rev_prev = {v: k for k, v in self.prev_ind_dict.items()}
        if ind_name in rev_prev.keys():
            old_indexing = rev_prev[ind_name]
            if old_indexing in self.rev_ind_map_to_subsampled.keys():
                new_index = self.rev_ind_map_to_subsampled[old_indexing]
            else:
                new_index = -1
        else:
            new_index = -1
        return new_index

    def tokenize_job(self, job):
        ipdb.set_trace()
        word_list = []
        for num, word in enumerate(job):
            if num < self.max_len - 3:
                word_list.append(word)
        return " ".join(word_list)

    def get_uniform_experience(self, career_len):
        return [round(i) for i in np.linspace(0, self.exp_levels - 1, career_len)]

    def check_monotonicity(self):
        all_labels = [i["exp_index"] for i in self.tuples]
        for cnt, user in enumerate(tqdm(self.user_lookup.keys(), desc=f"Checking monotonicity of experience for users...")):
            current_user = self.user_lookup[user]
            exp_seq_init = []
            for job in range(current_user[0], current_user[1]):
                exp_seq_init.append(all_labels[job])
            assert all(exp_seq_init[i] <= exp_seq_init[i + 1] for i in range(len(exp_seq_init) - 1))
        print("all experience are monotonic")
