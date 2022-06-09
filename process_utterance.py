from ltp import LTP
import pandas as pd
from tqdm import tqdm
import re
import configparser
key_dic = {}

class Process():
    def __init__(self, mode='rule'):
        assert mode in ['rule','pos','simple'], print('mode not in mode list')
        config = configparser.ConfigParser()
        config.read("./paths.cfg")
        self.mode = mode
        self.ltp = LTP(config['path']['LTP'])
        self.remain_d = ['没', '不']
        self.key_pos = ['a', 'i', 'v']
        self.ignore_verb = ['是','想','知道','觉得','要','有','感觉']
        self.pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|\？|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|；|·|！| |…'
        self.pos_pattern = [['v','v'],['v','n'], ['v','i'], ['v','u','z'],['v','u','m'],['v','c','v'],['v','c','i'],['a','v'],['v', 'u', 'd', 'd', 'a']]
        self.min_rebuild_layer = 1
        self.min_rebuild_entity = 2
        self.all_pos_entity = []
        self.all_dep_tree = []
        self.all_seg = []
        self.all_pos = []
        self.all_event = []
        self.all_event_str = []

    def extract_event_pos(self, sentence):
        seg_list, hidden = self.ltp.seg(sentence)
        pos_list = self.ltp.pos(hidden)
        all_entity = []
        for i, (pos, seg) in enumerate(zip(pos_list, seg_list)):
            all_entity.append([])
            def parse_entity(pattern):
                length = len(pattern)
                if len(pos) < length:
                    return []
                ret = []
                for i in range(len(pos)-length):
                    if pos[i:i+length] == pattern:
                        ret.append(''.join(seg[i:i+length]))
                return ret
            for pattern in self.pos_pattern:
                all_entity[i].extend(parse_entity(pattern))
        self.all_pos_entity.append(all_entity)

    def parse_dependency(self, sentence):
        seg_list, hidden = self.ltp.seg(sentence)
        pos_list = self.ltp.pos(hidden)
        dep_list = self.ltp.dep(hidden)
        for dep,seg,pos in zip(dep_list, seg_list, pos_list):
            count, epoch = 0, 0
            dep_tree = []
            root_idx = set()
            while 1:
                # print(dep_tree)
                dep_tree.append({})
                if epoch == 0:
                    hed = [i for i in dep if i[2] == 'HED' or i[1] == 0]
                    new_root_idx = set([i[0]-1 for i in hed])
                    while 1:
                        new_hed = [i for i in dep if i[2] == 'COO' and i[1]-1 in new_root_idx]
                        prev_length = len(hed)
                        hed.extend(new_hed)
                        current_length = len(hed)
                        new_root_idx = set([i[0]-1 for i in new_hed])
                        if prev_length == current_length:
                            break
                    root_idx = set([i[0]-1 for i in hed])
                    key = [i[0]-1 for i in hed]
                    value = [{'word':seg[k], 'pos': pos[k],'children':[], 'relation': []} for k in key]
                    dep_tree[0] = dict(zip(key, value))
                else:
                    upper_idx = set(dep_tree[epoch-1].keys())
                    current_node = [(i[0]-1,i[1]-1) for i in dep if (i[1]-1 in upper_idx) and not (i[1]-1 in root_idx and (i[2] in ['COO','HED']))]
                    key = [i[0] for i in current_node]
                    value = [{'word':seg[k], 'pos': pos[k], 'children':[],'relation':[]} for k in key]
                    dep_tree[epoch] = dict(zip(key, value))
                    for current_idx, upper_idx in current_node:
                        # 去掉标点和语气词
                        if dep[current_idx][2] != 'WP':
                            dep_tree[epoch-1][upper_idx]['children'].append(current_idx)
                            dep_tree[epoch-1][upper_idx]['relation'].append(dep[current_idx][2])
                epoch += 1
                count += len(key)
                if count == len(dep):
                    def rebuild(depth, idx):
                        if depth == 0:
                            new_key = {key:dep_tree[depth+1][key] for key in dep_tree[depth][idx]['children'] if not any(iv in dep_tree[depth+1][key]['word'] for iv in self.ignore_verb)} 
                            new_key_temp = new_key.copy()
                            for value in new_key.values():
                                for k,v in {idx:dep_tree[depth+2][idx] for idx, rel in zip(value['children'],value['relation']) if rel == 'COO' and not any(iv in dep_tree[depth+2][idx]['word'] for iv in self.ignore_verb)}.items():
                                    new_key_temp[k] = v
                                    index = value['children'].index(k)
                                    value['children'] = [c for c in value['children'] if c != k]
                                    value['relation'] = [value['relation'][i] for i in range(len(value['relation'])) if i != index]
                            new_key = new_key_temp
                            new_dep_tree_temp.append(new_key)
                            for value in new_dep_tree_temp[0].values():
                                for idx in value['children']:
                                    rebuild(depth+1, idx)
                        else:
                            if len(new_dep_tree_temp) == depth:
                                new_dep_tree_temp.append({})
                            for dic in dep_tree:
                                for k, v in dic.items():
                                    if k==idx:
                                        new_dep_tree_temp[depth][k] = v
                                        for idx in v['children']:
                                            rebuild(depth+1,idx)
                                        break
                    
                    new_dep_tree = []
                    for key ,value in dep_tree[0].items():
                        def count_key(key):
                            # 判断核心词连接的动词数
                            index = [idx for idx in dep_tree[0][key]['children']]
                            count_direct = sum([1 if dep_tree[1][idx]['pos'] in ['v','i'] and not any(iv in dep_tree[1][idx]['word'] for iv in self.ignore_verb) else 0 for idx in index])
                            count_coo = sum([1 if dep[i][1]-1 in index and dep[i][2] == 'COO' and pos[i] in ['v','i']  and not any(iv in seg[i] for iv in self.ignore_verb) else 0 for i in range(len(dep))])
                            count = count_direct+count_coo
                            return count
                        def judge_depth(key, layer):
                            global layer_num, entity_num
                            # 判断核心词下的动词引导的子树层数
                            if layer == 0:
                                layer_num = 0
                                entity_num = 0
                                # 第一层只计算动词
                                for child_idx in dep_tree[layer][key]['children']:
                                    if dep_tree[layer+1][child_idx]['pos'] in ['v','i'] and not any(iv in dep_tree[layer+1][child_idx]['word'] for iv in self.ignore_verb):
                                        judge_depth(child_idx, layer+1)
                                return layer_num, entity_num
                            elif layer==1:
                                # 第二层只计算动词后面的内容
                                for i,child_idx in enumerate(dep_tree[layer][key]['children']):
                                    def if_search(i):
                                        # print(key,i)
                                        # print(pos[dep_tree[layer][key]['children'][i]] == 'd')
                                        return dep_tree[layer][key]['children'][i] > key or (dep_tree[layer][key]['relation'][i] == 'ADV' and (pos[dep_tree[layer][key]['children'][i]] in ['v', 'a'] or (pos[dep_tree[layer][key]['children'][i]] == 'd' and any(d in dep_tree[layer+1][dep_tree[layer][key]['children'][i]]['word'] for d in ['没','不']))))
                                    if if_search(i):
                                        entity_num += 1
                                        layer_num = max(layer_num,layer)
                                        judge_depth(child_idx, layer+1)
                            else:
                                # 其他层全部计算
                                for child_idx in dep_tree[layer][key]['children']:
                                    entity_num += 1
                                    layer_num = max(layer_num,layer)
                                    judge_depth(child_idx, layer+1)
                        new_dep_tree_temp = []
                        if (any(verb in value['word'] and value['pos']=='v' for verb in self.ignore_verb) or count_key(key) > 1):
                            new_key = [verb_child for verb_child in dep_tree[0][key]['children'] if dep_tree[1][verb_child]['pos'] in ['i','v']]
                            new_key_coo = []
                            for new_k in new_key:
                                new_key_coo.extend(dep_tree[1][new_k]['children'][i] for i in range(len(dep_tree[1][new_k]['children'])) if dep_tree[1][new_k]['relation'][i] == 'COO')
                            new_key.extend(new_key_coo)
                            if new_key == []:
                                continue
                            layer_num, entity_num = judge_depth(key,0)
                            if layer_num >= self.min_rebuild_layer and entity_num/len(new_key) >= self.min_rebuild_entity:
                                rebuild(0, key)
                                remain_idx = min(len(new_dep_tree), len(new_dep_tree_temp))
                                new_dep_tree_t = [{**new_dep_tree[i], **new_dep_tree_temp[i]} for i in range(remain_idx)]
                                new_dep_tree_t.extend(new_dep_tree[remain_idx:] if len(new_dep_tree)>len(new_dep_tree_temp) else new_dep_tree_temp[remain_idx:])
                                new_dep_tree = new_dep_tree_t
                    if new_dep_tree != []:
                        dep_tree = new_dep_tree
                    self.all_dep_tree.append(dep_tree)
                    self.all_seg.append(seg)
                    self.all_pos.append(pos)
                    break
    
    def extract_event_dep(self):
        for dep_tree, pos in zip(self.all_dep_tree, self.all_pos):
            event_list = []

            def DFS(depth, event = None, idx = None):
                if depth == 0:
                    count = -1
                    for key, value in dep_tree[0].items():
                        if pos[key] in ['v','i']:
                            def if_search(i):
                                return value['children'][i] > key or (value['relation'][i] == 'ADV' and (pos[value['children'][i]] in ['v', 'a'] or (pos[value['children'][i]] == 'd' and any(d in dep_tree[1][value['children'][i]]['word'] for d in ['没','不']))))
                        elif pos[key] == 'a':
                            def if_search(i):
                                return value['children'][i] < key and (value['relation'][i] == 'SBV' or (pos[value['children'][i]] == 'd' and any(d in dep_tree[1][value['children'][i]]['word'] for d in ['没','不'])))
                        else:
                            continue
                        count += 1
                        search_idx = [value['children'][i] for i in range(len(value['children'])) if if_search(i)]
                        event_list.append([key])
                        for s_idx in search_idx:
                            DFS(depth+1, event_list[count], s_idx)
                else:
                    key = idx
                    event.append(idx)
                    for s_idx in dep_tree[depth][key]['children']:
                        DFS(depth+1, event, s_idx)

            DFS(0)
            self.all_event.append(event_list)
            
    
    def filter_event(self, sub_utterance_ask, position_list):
        all_event = []
        ask_pointer = 0
        state_pointer = 0
        for i, position_info in enumerate(position_list):
            all_event.append([])
            if position_info == 'ask':
                all_event[i].append(sub_utterance_ask[ask_pointer])
                ask_pointer += 1
            else:
                event, seg = self.all_event[state_pointer], self.all_seg[state_pointer]
                for sub_event in event:
                    sub_event.sort()
                    sub_event_str = ''.join([seg[id] for id in sub_event])
                    all_event[i].append(sub_event_str)
                state_pointer += 1
        self.all_event_str.append(all_event)

    def preprocess(self, item):
        sentence_list = re.split(self.pattern, item)
        notation_list = re.findall(self.pattern, item)
        result_list = [sentence+notation for sentence,notation in zip(sentence_list, notation_list)]
        if result_list == []:
            result_list = [item]
        try:
            result_list = [result if result[-1]!='？' and result[-1]!= '?' else '有人问'+result for result in result_list]
        except:
            print(result_list)
        result_list = [result for result in result_list if len(result)>=5]
        def position_info(result):
            if result[-1]!='？' and result[-1]!= '?':
                return 'state'
            else:
                return 'ask'
        position_list = [position_info(result) for result in result_list]
        if len(result_list) > 0:
            result_list = [result[:-1] for result in result_list[:-1]] + [result_list[-1]]
        result_list_state = [result_list[i] for i in range(len(result_list)) if position_list[i] == 'state']
        result_list_ask = [result_list[i] for i in range(len(result_list)) if position_list[i] == 'ask']
        if self.mode == 'rule':
            return position_list, result_list_state, result_list_ask
        else:
            return result_list


    def __call__(self, sentences):
        if self.mode == 'rule':
            for sentence in tqdm(sentences):
                self.all_event = []
                self.all_dep_tree = []
                self.all_seg = []
                self.all_pos = []
                self.had_pos = {}
                position_list, sub_utterance_state, sub_utterance_ask = self.preprocess(sentence)
                if sub_utterance_state == []:
                    self.all_event_str.append([sub_utterance_ask])
                    continue
                self.parse_dependency(sub_utterance_state)
                self.extract_event_dep()
                self.filter_event(sub_utterance_ask, position_list)
            return self.all_event_str
        elif self.mode == 'pos':
            for sentence in tqdm(sentences):
                self.extract_event_pos([sentence])
            return self.all_pos_entity
        else:
            for sentence in tqdm(sentences):
                result_list = self.preprocess(sentence)
                self.all_event_str.append([[r_l] for r_l in result_list])
            return self.all_event_str
