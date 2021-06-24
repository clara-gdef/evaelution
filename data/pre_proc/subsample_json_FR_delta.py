import os
import argparse
import pickle as pkl
import ipdb
import yaml
from tqdm import tqdm
import re
import pandas as pd
import fasttext
from nltk.tokenize import word_tokenize
import json
from datetime import datetime


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        for split in ["TEST", "VALID", "TRAIN"]:
            build_dataset(args, split)


def build_dataset(args, split):
    current_file = os.path.join(CFG["gpudatadir"], f'{CFG["json_file"]}_{split}.json')
    language_classifier = fasttext.load_model(os.path.join(CFG["modeldir"], "lid.176.bin"))

    with open(current_file, 'r') as f:
        num_lines = sum(1 for line in f)
    with open(current_file, 'r') as f:
        pbar = tqdm(f, total=num_lines)
        dataset = []
        for line in pbar:
            current_person = json.loads(line)
            jobs = current_person[1]
            if len(jobs) >= args.MIN_JOB_COUNT:
                new_p = build_new_person(current_person, language_classifier)
                if len(new_p) > 2 and len(new_p[-1]) >= args.MIN_JOB_COUNT:
                    dataset.append(new_p)
                    if len(dataset) == 100:
                        ppl_file = os.path.join(CFG["gpudatadir"], f"profiles_jobs_ind_title_delta_years_{split}_100.pkl")
                        with open(ppl_file, 'wb') as fp:
                            pkl.dump(dataset, fp)
                        print("Toy dataset saved at " + ppl_file)
            pbar.update(1)

    ppl_file = os.path.join(CFG["gpudatadir"], f"profiles_jobs_ind_title_delta_years_{split}.pkl")
    with open(ppl_file, 'wb') as fp:
        pkl.dump(dataset, fp)


def build_new_person(person, language_classifier):
    person_id = person[0]
    industry = person[-1]
    new_p = [person_id, industry]
    jobs = []
    for job in person[1]:
        end = handle_date(job)
        tstmp = pd.Timestamp.fromtimestamp(job["from_ts"])
        start = round(datetime.timestamp(tstmp.round("D").to_pydatetime()))
        if (end > 0) and (start > 0):  # corresponds to the timestamp of 01/01/1970
            new_j = job["position"] + " " + job["description"]
            predicted_lang = identify_language(job, language_classifier)
            if (predicted_lang[0][0][0] == "__label__fr" or predicted_lang[0][0][0] == "__label__en") and (predicted_lang[1][0][0] > .6) and (len(job) >= args.min_len):
                delta = round(
                    (datetime.fromtimestamp(end) - datetime.fromtimestamp(start)).days / 365)
                if delta >= 0:
                    j = {'from': start,
                         'to': end,
                         'job': new_j,
                         'delta_years': delta}
                    jobs.append(j)
    if len(jobs) >= args.MIN_JOB_COUNT:
        new_p.append(jobs)
    return new_p

def handle_date(job):
    if job["to"] == "Present":
        date_time_str = '2018-04-12'  # date of files creation
        time = datetime.timestamp(datetime.strptime(date_time_str, '%Y-%m-%d'))
    elif len(job["to"].split(" ")) == 2:
        try:
            time = datetime.timestamp(datetime.strptime(job["to"], "%B %Y"))
        except ValueError:
            time = datetime.timestamp(datetime.strptime(job["to"].split(" ")[-1], "%Y"))
    else:
        try:
            time = datetime.timestamp(datetime.strptime(job["to"].split(" ")[-1], "%Y"))
        except ValueError:
            date_time_str = '2018-04-13'  # date of files creation
            time = datetime.timestamp(datetime.strptime(date_time_str, '%Y-%m-%d'))
    tstmp = pd.Timestamp.fromtimestamp(time)
    return round(datetime.timestamp(tstmp.round("D").to_pydatetime()))


def identify_language(job, ft_model):
    jobs_str = " ".join(job)
    ft_model.predict([jobs_str])
    return ft_model.predict([jobs_str])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--MIN_JOB_COUNT", type=int, default=3)
    args = parser.parse_args()
    main(args)
