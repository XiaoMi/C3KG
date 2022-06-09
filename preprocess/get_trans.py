import uuid
import json
import requests
import pandas as pd
from tqdm import tqdm
import configparser


def request_dev(query):
    # rewrite using any translation model or API
    raise NotImplementedError("rewrite using any translation model or API")


if __name__ == '__main__':
    print('Translating...')
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    tail_translated = []
    head_translated = []
    
    tail = pd.read_csv(config["path"]["unique_tail_replaced"],  usecols=['tail', 'tail_replaced'], sep='\t')
    for h in tqdm(tail['tail_replaced']):
        tail_translated.append(request_dev(h))
    tail['tail_translated'] = tail_translated
    tail.to_csv(config["path"]["unique_tail_replaced_translated"],header=None, index=None, encoding="utf_8_sig")

    head = pd.read_csv(config["path"]["unique_head_replaced"],  usecols=['head', 'head_replaced'], sep='\t')
    for h in tqdm(head['head_replaced']):
        head_translated.append(request_dev(h))
    head['head_translated'] = head_translated
    head.to_csv(config["path"]["unique_head_replaced_translated"],header=None, index=None, encoding="utf_8_sig")
