import configparser
config = configparser.ConfigParser()
config.read("./paths.cfg")
import pickle
from sentence_transformers import SentenceTransformer, util
import os
import pandas as pd
chosen_relation = set(['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oWant', 'oReact','isBefore','isAfter','HasSubEvent'])

def build_new_table(*filenames):
    '''
    建立head——>tail映射表
    '''
    if not os.path.exists(config["path"]["head2tail"]):
        print('开始建立head2tail映射表')
        head2tail = {}
        for filename in filenames:
            df = pd.read_csv(filename, sep='\t')
            for _, item in df.iterrows():
                head = item['head']
                relation = item['relation']
                tail = item['tail']
                if head not in head2tail.keys():
                    head2tail[head] = {}
                if relation not in head2tail[head].keys():
                    head2tail[head][relation] = []
                head2tail[head][relation].append(tail)
        print('Done')
        with open(config["path"]["head2tail"],'wb')as f:
            pickle.dump(head2tail,f)
        return head2tail
    else:
        with open(config["path"]["head2tail"],'rb')as f:
            head2tail = pickle.load(f)
            return head2tail


def append_tail(item, head2tail, filter_relation=True):
    '''
    传入一个head，找到全部对应的relation和tail返回
    '''
    ret = {}
    for head_similarity in item:
        for head, similarity in head_similarity:
            ret[head] = {}
            ret[head]['similarity'] = similarity
            for relation in head2tail[head].keys():
                if not filter_relation or relation in chosen_relation:
                    ret[head][relation] = head2tail[head][relation]
    return ret
                    

def match(head_embeddings, query_embeddings, trg):
    '''
    根据语义相似度进行匹配，返回每个子句相似度最高的head
    '''
    all_matched = []
    for q_embedding in query_embeddings:
        matched = []
        cosine_scores = util.pytorch_cos_sim(q_embedding, head_embeddings).squeeze()
        values, indices = cosine_scores.topk(1, dim=0, largest=True, sorted=True)
        for value,indice in zip(values,indices):
            matched.append((trg[indice.item()], value.item()))
        all_matched.append(matched)
    return all_matched
