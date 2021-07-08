import argparse
from collections import Counter

import ipdb
import yaml
import numpy as np
from tqdm import tqdm

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

        sub_user_train, sub_tup_train = subsample_users_by_career_len(args, train_lu, data_train.tuples, args.iteration, "train")
        sub_user_valid, sub_tup_valid = subsample_users_by_career_len(args, valid_lu, data_valid.tuples, args.iteration, "valid")
        sub_user_test, sub_tup_test = subsample_users_by_career_len(args, test_lu, data_test.tuples, args.iteration, "test")

        new_users_train, exp_seq = turn_exp_sequence_into_jump_seq(args, sub_user_train, sub_tup_train, exp_seq, "train")
        new_users_valid, exp_seq = turn_exp_sequence_into_jump_seq(args, sub_user_valid, sub_tup_valid, exp_seq, "valid")
        new_users_test, exp_seq = turn_exp_sequence_into_jump_seq(args, sub_user_test, sub_tup_test, exp_seq, "test")

        exp_seq, carreer_len = get_exp_sequence(sub_user_train, sub_tup_train, exp_seq, carreer_len, args.iteration,
                                                args.mod_type, "train")
        exp_seq, carreer_len = get_exp_sequence(sub_user_valid, sub_tup_valid, exp_seq, carreer_len, args.iteration,
                                                args.mod_type, "valid")
        exp_seq, carreer_len = get_exp_sequence(sub_user_test, sub_tup_test, exp_seq, carreer_len, args.iteration,
                                                args.mod_type, "test")
        total_users = len(train_lu) + len(valid_lu) + len(test_lu)
        exp_seq.most_common(10)
        prct_exp_seq = [(i, 100 * v / total_users) for i, v in exp_seq.most_common(10)]
        prct_exp_seq_seq = [i[0] for i in prct_exp_seq]
        prct_exp_seq_value = [str(i[1]) for i in prct_exp_seq]
        print('\n'.join(prct_exp_seq_seq))
        print('\n'.join(prct_exp_seq_value))
        prct_career_len = [(i, 100 * v / total_users) for i, v in carreer_len.most_common(10)]
        ipdb.set_trace()


def turn_exp_sequence_into_jump_seq(args, users, tuples, exp_seq, split):
    new_users = {}
    for num, user in enumerate(tqdm(users, desc=f"parsing users for {split} split...")):
        current_user = users[user]
        start = current_user[0]
        end = current_user[1]
        user_rep = np.zeros(args.max_career_len)
        prev_exp = tuples[start][-1]
        for num, job in enumerate(range(start+1, end)):
            current_exp = tuples[job][-1]
            if current_exp > prev_exp:
                user_rep[num] = 1.
            prev_exp = current_exp
        new_users[user] = user_rep
        exp_seq[int(sum(new_users))] += 1
    ipdb.set_trace()
    return new_users, exp_seq


def get_exp_sequence(users, jobs, exp_seq, carreer_len, iteration, mod_type, split):
    for num, user in enumerate(tqdm(users, desc=f"parsing users for {split} split...")):
        current_user = users[user]
        start = current_user[0]
        end = current_user[1]
        current_seq = []
        for job in range(start, end):
            if split == "test" or iteration == 0:
                tmp = jobs[job]["exp_index"]
            else:
                tmp = jobs[job][-1]
            current_seq.append(tmp)
        if len(current_seq) > 0:
            carreer_len[len(current_seq)] += 1
            exp_seq[str(current_seq)] += 1
        ipdb.set_trace()
    print(f"Number of sequence acquired: {sum([i for i in exp_seq.values()])}")
    return exp_seq, carreer_len


def subsample_users_by_career_len(args, users, jobs, iteration, split):
    retained_users = {}
    retained_jobs = []
    cnt_strt, cnt_end = 0, 0
    for num, user in enumerate(tqdm(users, desc=f"parsing users for {split} split...")):
        current_user = users[user]
        start = current_user[0]
        end = current_user[1]
        num_jobs = end - start
        if num_jobs >= args.min_career_len:
            cnt_end = min(start + num_jobs, start + args.max_career_len)
            retained_users[user] = [cnt_strt, cnt_end]
            for job in range(start, cnt_end):
                if split == "test" or iteration == 0:
                    tmp = jobs[job]["exp_index"]
                else:
                    tmp = jobs[job][-1]
                retained_jobs.append(tmp)
            cnt_strt = cnt_end
    print(f"Retained users : {100*len(retained_users)/len(users)} %")
    print(f"Retained jobs : {100*len(retained_jobs)/len(jobs)} %")
    return users, jobs



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
    parser.add_argument("--mod_type", type=str, default="ft")  # nb or ft
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--enforce_monotony", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="iter")
    parser.add_argument("--min_career_len", type=int, default=3)
    parser.add_argument("--max_career_len", type=int, default=8)
    args = parser.parse_args()
    main(args)
