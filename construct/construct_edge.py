import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import json
import networkx as nx
from tqdm import tqdm
utterance_window_size = 1
sentence_window_size = 1
entity_window_size = 1
emotion_window_size = 30
DG = nx.DiGraph()
chosen_relation = set(['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oWant', 'oReact','isBefore','isAfter','HasSubEvent'])
before_relation = set(['isAfter','xNeed'])
after_relation1 = set(['xEffect','xWant', 'xIntent','isBefore'])
after_relation2 = set(['oReact','oWant'])
tail2label_table = None
import configparser
from ltp import LTP



def stat():
    data = json.load(open('../data/多轮对话数据_match_result.json',encoding='utf8'))
    match_count = 0
    total_count = 0
    for dialog in data['data']:
        match_list = [1 if dic != {} else 0 for dic in dialog['match_result']]
        match_count += sum(match_list)
        total_count += len(match_list)
    print(match_count, total_count)

def stat_graph():
    next_utterance_dict = {}
    next_sentence_dict = {}
    next_entity_dict = {}
    emotion_reasoning_flow_dict = {}
    emotion_empathy_flow_dict = {}
    origin_count = 0
    G = nx.read_gpickle('MOD_windowSize1.Graph')
    head_set = set()
    for e in G.edges(data=True):
        attr = e[2]
        if attr['type'] == 'next_utterance':
            if attr['weight'] not in next_utterance_dict.keys():
                next_utterance_dict[attr['weight']] = 0
            next_utterance_dict[attr['weight']] += 1
            head_set.add(e[0])
        elif attr['type'] == 'next_entity' :
            if attr['weight'] not in next_entity_dict.keys():
                next_entity_dict[attr['weight']] = 0
            next_entity_dict[attr['weight']] += 1
            head_set.add(e[0])
        elif attr['type'] == 'next_sentence':
            if attr['weight'] not in next_sentence_dict.keys():
                next_sentence_dict[attr['weight']] = 0
            next_sentence_dict[attr['weight']] += 1
            head_set.add(e[0])
        elif attr['type'] == 'emotion_cause':
            if attr['weight'] not in emotion_reasoning_flow_dict.keys():
                emotion_reasoning_flow_dict[attr['weight']] = 0
            emotion_reasoning_flow_dict[attr['weight']] += 1
        # elif attr['type'][:6] == 'intent':
        elif attr['type'] == 'emotion_intent':
            if attr['weight'] not in emotion_empathy_flow_dict.keys():
                emotion_empathy_flow_dict[attr['weight']] = 0
            emotion_empathy_flow_dict[attr['weight']] += 1
        else:
            origin_count += 1
            head_set.add(e[0])
    print('next_utterance:',sum(next_utterance_dict.values()), 'next_sentence:',sum(next_sentence_dict.values()), 'next_entity:',sum(next_entity_dict.values()), 'emotion_reasoning_flow:',sum(emotion_reasoning_flow_dict.values()),'emotion_empathy_flow:',sum(emotion_empathy_flow_dict.values()))
    next_utterance_dict, next_sentence_dict, next_entity_dict, emotion_reasoning_flow_dict, emotion_empathy_flow_dict = sorted(next_utterance_dict.items(), key=lambda x: x[0]), sorted(next_sentence_dict.items(), key=lambda x: x[0]), sorted(next_entity_dict.items(), key=lambda x: x[0]), sorted(emotion_reasoning_flow_dict.items(), key=lambda x: x[0]),sorted(emotion_empathy_flow_dict.items(), key=lambda x: x[0])
    print('next_sentence',next_sentence_dict)
    print('next_utterance',next_utterance_dict)
    print('next_entity',next_entity_dict)
    print('emotion_cause_flow',emotion_reasoning_flow_dict)
    print('emotion_intent_flow',emotion_empathy_flow_dict)
    print('origin_ATOMIC_flow', origin_count)
    print(len(G.nodes))
    print('origin head node:',len(head_set))

def create_edge(head1,head2,type):
    # DG.add_node_from([head1,head2])
    # print(head1,head2)
    if DG.has_edge(head1,head2) and DG.adj[head1][head2]['type'] == type:
        DG.adj[head1][head2]['weight'] += 1
    else:
        DG.add_edge(head1, head2, weight=1, type=type)

def build_next_utterance(src, trg):
    global DG
    for head1, _ in src.items():
        for head2, _ in trg.items():
            create_edge(head1, head2,'next_utterance')

def build_next_sentence(src):
    global DG
    for i, (head1,_) in enumerate(src.items()):
        current_sentence_window_size = min(sentence_window_size+1, len(src)-i)
        # print(current_sentence_window_size)
        for j in range(1,current_sentence_window_size):
            head2 = list(src.keys())[i+j]
            # print(head1,head2)
            create_edge(head1, head2, 'next_sentence')

def build_next_entity(src):
    global DG
    for i, entity1 in enumerate(src):
        current_entity_window_size = min(entity_window_size+1, len(src)-i)
        for j in range(1,current_entity_window_size):
            entity2 = src[i+j]
            create_edge(entity1,entity2, 'next_entity')


def extract_keyword(content):
    try:
        seg, hidden = ltp.seg(content)
    except:
        return []
    pos = ltp.pos(hidden)
    keyword_list = []
    for i,p in enumerate(pos):
        keyword_list.append([seg[i][j] for j, tag in enumerate(p) if tag in ['v','a','n']])
    return keyword_list

def tail2label(tails):
    global tail2label_table
    if tail2label_table == None:
        tail2label_table = json.load(open(config["path"]["Tail2Emotion"],encoding='utf8'))
    tail_labels = []
    for tail in tails:
        if tail in tail2label_table.keys():
            tail_labels.append(tail2label_table[tail])
        else:
            tail_labels.append(-1)
    return tail_labels


def build_emotion_flow(match_result, content, emotion_labels, intent_labels=None):
    C2E = {'询问':'ask','安慰':'console','描述':'describe','观点':'opinion','others':'others','建议':'advise'}
    content_keyword = extract_keyword(content)
    for i,result in enumerate(match_result):
        if emotion_labels[i] == 'others':
            continue
        for _, sim_tail in result.items():
            emotion_choosen = []
            relation_action = []
            for rel, tails in sim_tail.items():
                if rel == 'similarity':
                    continue
                if rel in ['xAttr','xReact']:
                    tail_label = tail2label(tails)
                    emotion_choosen.extend([tail for j,tail in enumerate(tails) if tail_label[j] == emotion_labels[i]])
            if len(emotion_choosen) == 0:
                continue
            for rel, tails in sim_tail.items():
                if rel in before_relation:
                    current_emotion_window_size = min(emotion_window_size, i)
                    c_keyword = [content_keyword[j] for j in range(i-current_emotion_window_size, i)]
                    c_relation = ['emotion_cause' for j in range(i-current_emotion_window_size, i)]
                elif rel in after_relation1:
                    current_emotion_window_size = min(len(content)-i-1, emotion_window_size)
                    c_keyword = [content_keyword[j] for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==0]
                    if intent_labels != None:
                        c_relation = ['intent_'+C2E[intent_labels[j]] for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==0]
                    else:
                        c_relation = ['emotion_intent' for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==0]
                elif rel in after_relation2:
                    current_emotion_window_size = min(len(content)-i-1, emotion_window_size)
                    c_keyword = [content_keyword[j] for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==1]
                    if intent_labels != None:
                        c_relation = ['intent_'+C2E[intent_labels[j]] for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==1]
                    else:
                        c_relation = ['emotion_intent' for j in range(i+1,i+current_emotion_window_size) if (j-i)%2==1]
                else:
                    continue
                for tail in tails:
                    tail_keyword = extract_keyword([tail])[0]
                    for con_keyword, relation in zip(c_keyword, c_relation):
                        for t_keyword in tail_keyword:
                            if t_keyword in con_keyword:
                                relation_action.append((relation,tail))
                                break
            all_edge = []
            for emotion in emotion_choosen:
                all_edge.extend([(emotion,action,relation) for (relation,action) in relation_action])
            all_edge = set(all_edge)
            for edge in all_edge:
                print(edge,content[i])
                create_edge(edge[0],edge[1],edge[2])

def build_head2tail_relation(*head2tail):
    for match_result in head2tail:
        for i in range(len(match_result)):
            for head, sim_tail in match_result[i].items():
                for relation, tails in sim_tail.items():
                    if relation == 'similarity':
                        continue
                    for tail in tails:
                        create_edge(head,tail,relation)


def build_in_MOD():
    emotion2id = json.load(open('emotion_dict.json'))
    id2emotion = {}
    for k, v in emotion2id.items():
        id2emotion[v] = k
    label2emotion = json.load(open('MODEmotionLabelMatch.json'))
    emotion2label = {}
    for k, v in label2emotion.items():
        for value in v:
            emotion2label[value] = k
    global DG
    data = json.load(open(config["path"]["MOD_match"],encoding='utf8'))
    for _, dialog in tqdm(data.items()):
        all_match_result = []
        all_dialog_content = []
        all_emotion_labels = []
        all_entity_match_result = []
        for i,utterance in enumerate(dialog):
            current_utterance_window_size = min(utterance_window_size+1, len(dialog)-i)
            build_next_sentence(utterance['match_result'])
            for j in range(1,current_utterance_window_size):
                build_next_utterance(utterance['match_result'],dialog[i+j]['match_result'])
                entity_list = []
            all_dialog_content.append(utterance['txt'])
            all_match_result.append(utterance['match_result'])
            if 'emotion_id' in utterance.keys() and id2emotion[utterance['emotion_id']] in emotion2label.keys():
                all_emotion_labels.append(emotion2label[id2emotion[utterance['emotion_id']]])
            else:
                all_emotion_labels.append('others')
            all_entity_match_result.append(utterance['entity_matched_result'])
        entity_list.extend([key for key in utterance['entity_matched_result'].keys()])
        build_next_entity(entity_list)
        build_head2tail_relation(all_match_result,all_entity_match_result)
        build_emotion_flow(all_match_result,all_dialog_content,all_emotion_labels)
    nx.write_gpickle(DG, config["path"]["MOD_KG"])




def main():
    global DG

    data = json.load(open(config["path"]["Cconv_matched"],encoding='utf8'))
    for dialog in tqdm(data['data']):
        event_match_result = dialog['match_result']
        entity_matched_result = dialog['entity_matched_result']
        build_head2tail_relation(event_match_result,entity_matched_result)
        for i in range(len(event_match_result)):
            current_utterance_window_size = min(utterance_window_size+1, len(event_match_result)-i)
            build_next_sentence(event_match_result[i])
            for j in range(1,current_utterance_window_size):
                build_next_utterance(event_match_result[i],event_match_result[i+j])
                entity_list = []
        for entity_dic in entity_matched_result:
            entity_list.extend([key for key in entity_dic.keys() if key != 'similarity'])
            build_next_entity(entity_list)

    data = json.load(open(config["path"]["emotion_matched"],encoding='utf8'))
    for dialog in tqdm(data):
        all_match_result = []
        all_dialog_content = []
        all_emotion_labels = []
        all_intent_labels = []
        for utterance in dialog:
            all_dialog_content.append(utterance['sens'])
            all_match_result.append(utterance['match_result'])
            all_emotion_labels.append(utterance['label'])
            all_intent_labels.append(utterance['intent'])
        build_emotion_flow(all_match_result,all_dialog_content,all_emotion_labels, all_intent_labels)
    nx.write_gpickle(DG, config["path"]["C3KG"])


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    ltp = LTP(config['path']['LTP'])
    main()
