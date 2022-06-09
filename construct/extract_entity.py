import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ltp import LTP
import pandas as pd
from tqdm import tqdm
import re
import json
import configparser
head2tail = None
from utils import *
head2tail = build_new_table(config["path"]["ATOMIC_Chinese"])

def append_tail(item):
    '''
    传入一个head，找到全部对应的relation和tail返回
    '''
    ret = {}
    for head in item:
        if head in head2tail.keys():
            ret[head] = {}
            for relation in head2tail[head].keys():
                ret[head][relation] = head2tail[head][relation]
    return ret


def match(src, trg):
    ret = []
    for s in src:
        if s in trg:
            ret.append(s)
    return ret

class Extract():
    def __init__(self):
        self.ltp = LTP(config["path"]["LTP"])
        self.all_result = []

    def extract(self, sentence):
        seg, hidden = self.ltp.seg([sentence])
        pos = self.ltp.pos(hidden)
        result = []
        for i,pos_tag in enumerate(pos[0]):
            if pos_tag in ['a','v','n']:
                result.append(seg[0][i])
        self.all_result.append(result)

    def __call__(self, sentences):
        for sentence in tqdm(sentences):
            self.extract(sentence)
        return self.all_result

def process_all():
    build_new_table(config["path"]["ATOMIC_Chinese"])
    head = pd.read_csv(config["path"]["head_phrase"])
    trg = set(head['head_translated'])
    extract = Extract()
    data = json.load(open(config["path"]["Cconv_matched"],'r',encoding='utf8'))
    content = []
    for dialog in data['data']:
        content.extend(dialog['content'])
    all_entity_list = extract(content)

    i = 0
    for dialog in data['data']:
        dialog['entity_extract'] = all_entity_list[i:i+len(dialog['content'])]
        i += len(dialog['content'])
    assert i == len(content)

    all_entity_list = [append_tail(match(entity_list,trg)) for entity_list in all_entity_list]

    i = 0
    assert len(all_entity_list) == len(content)
    for dialog in data['data']:
        dialog['entity_matched_result'] = all_entity_list[i:i+len(dialog['content'])]
        i += len(dialog['content'])
    assert i == len(content)
    with open(config["path"]["Cconv_matched"], "w",encoding='utf8') as f:
        f.write(json.dumps(data, ensure_ascii=False))



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    process_all()
