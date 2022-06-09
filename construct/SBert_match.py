import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import pandas as pd
from tqdm import tqdm
import re
import json
import pickle
from process_utterance import Process
trg = None
# 选取的适合对话任务的一些relation
head2tail = None
import configparser
from utils import *


def process_EmotionLabel():
    model = SentenceTransformer(config["path"]["SBert-ATOMIC"])
    global trg,head2tail
    process = Process()
    head = pd.read_csv(config["path"]["head_shortSentence"])
    head2tail = build_new_table(config["path"]["ATOMIC_Chinese"])
    trg = list(head['head_translated'])
    head_embeddings = model.encode(trg, convert_to_tensor=True)
    data = json.load(open(config["path"]["Emotion"],'r',encoding='utf8'))
    content = []
    for dialog in data:
        for utterance in dialog:
            content.append(utterance['sens'])
    all_event_list = process(content)
    all_match_result = []
    assert len(content) == len(all_event_list)
    count = 0
    for i, event_list in enumerate(tqdm(all_event_list)):
        src = []
        for event in event_list:
            src.extend(event)
        if src == []:
            count += 1
            all_match_result.append({})
            continue
        query_embeddings = model.encode(src, convert_to_tensor=True)
        all_match_result.append(match(head_embeddings, query_embeddings, trg))
    i = 0
    assert len(all_match_result) == len(content)
    for dialog in data:
        for utterance in dialog:
            utterance['match_result'] = append_tail(all_match_result[i], head2tail)
            i += 1
    assert i == len(content)
    with open(config["path"]["emotion_matched"], "w",encoding='utf8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
    print(count)


def process_all():
    model = SentenceTransformer(config["path"]["SBert-ATOMIC"])
    global trg,head2tail
    process = Process(mode='rule')
    head = pd.read_csv(config["path"]["head_shortSentence"])                                                                                                                                                                                                                        
    head2tail = build_new_table(config["path"]["ATOMIC_Chinese"])
    trg = list(head['head_translated'])
    head_embeddings = model.encode(trg, convert_to_tensor=True)
    data = json.load(open(config["path"]["Cconv"],'r',encoding='utf8'))
    content = []
    for dialog in data['data']:
        content.extend(dialog['content'])
    all_event_list = process(content)
    assert len(all_event_list) == len(content)
    all_match_result = []
    for i, event_list in enumerate(tqdm(all_event_list)):
        src = []
        for event in event_list:
            src.extend(event)
        if src == []:
            all_match_result.append({})
            continue
        query_embeddings = model.encode(src, convert_to_tensor=True)
        all_match_result.append(append_tail(match(head_embeddings, query_embeddings, trg), head2tail))
    i = 0
    assert len(all_match_result) == len(content)
    for dialog in data['data']:
        dialog['match_result'] = all_match_result[i:i+len(dialog['content'])]
        i += len(dialog['content'])
    assert i == len(content)
    with open(config["path"]["Cconv_matched"], "w",encoding='utf8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
    


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    process_EmotionLabel()
    process_all()