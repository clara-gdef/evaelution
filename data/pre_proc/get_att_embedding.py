import ipdb
import argparse
import yaml
import os

import torch
import pickle as pkl
from transformers import CamembertTokenizer, CamembertModel


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
    ind_en = ['Computer Software', 'Internet', 'Banking', 'Mechanical or Industrial Engineering', 'Human Resources',
              'Research', 'Management Consulting', 'Government Administration', 'International Trade and Development',
              'Retail', 'Public Relations and Communications', 'Food Production', 'Construction', 'Automotive',
              'Professional Training & Coaching', 'Telecommunications', 'Renewables & Environment',
              'Aviation & Aerospace', 'Insurance', 'Pharmaceuticals']
    # translated with deep L
    tgt_ind_file = "20_industry_dict_fr.pkl"
    ind_fr = ["Logiciels", "Internet", "Banques", "Génie mécanique ou industriel", "Ressources humaines", "Recherche",
              "Conseil en gestion", "Administration publique", "Commerce international et développement",
              "Commerce de détail", "Relations publiques et communication", "Production alimentaire", "Construction",
              "Automobile", "Formation professionnelle et coaching", "Télécommunications",
              "Énergies renouvelables et environnement", "Aviation et aérospatiale", "Assurances",
              "Produits pharmaceutiques"].
    dict_fr = {num: name for num, name in enumerate(ind_fr)}
    with open(os.path.join(CFG["gpudatadir"], tgt_ind_file), 'wb') as f_name:
        pkl.dump(dict_fr, f_name)
    ipdb.set_trace()

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    text_encoder = CamembertModel.from_pretrained('camembert-base')

    ind_embeddings = torch.zeros(20, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    args = parser.parse_args()
    init(args)
