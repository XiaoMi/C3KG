import pandas as pd
import re
import configparser
joint_translated_table = {}

def build_translated_table(df, key_col, value_col):
    ret = {}
    key_list = list(df[key_col])
    value_list = list(df[value_col])
    ret = dict(zip(key_list,value_list))
    return ret

def build_joint_translated_table():
    print('Building joint translation table...')
    global joint_translated_table
    head_meaning = {}
    df = pd.read_csv("../data/difference.csv")
    for idx, item in df.iterrows():
        if item[0] not in joint_translated_table.keys():
            joint_translated_table[item[0]] = {}
        joint_translated_table[item[0]][item[2]] = {}
        if item[1] not in joint_translated_table[item[0]][item[2]].keys():
            joint_translated_table[item[0]][item[2]][item[4]] = {'head_translated': item[4], 'tail_translated': item[6]}
        if item[0] not in head_meaning.keys():
            head_meaning[item[0]] = {}
        if item[3] not in head_meaning[item[0]].keys():
            head_meaning[item[0]][item[3]] = 0
        if item[4] not in head_meaning[item[0]].keys():
            head_meaning[item[0]][item[4]] = 0
        head_meaning[item[0]][item[3]]+=1
        head_meaning[item[0]][item[4]]+=1
    for head, freq in head_meaning.items():
        max_freq = max([f for f in freq.values()])
        choosen_meaning = [meaning for meaning in freq.keys() if freq[meaning] == max_freq][0]
        for tail, result in joint_translated_table[head].items():
            result['head_translated'] = choosen_meaning

def filter_phrase():
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    df = pd.read_csv(config["path"]["unique_head_replaced_translated"], usecols=['head','head_replaced','head_translated'])
    px = re.compile(r'.*(Person.).*',re.I)
    withPerson_list = []
    noPerson_list = []
    for i,item in enumerate(df.iterrows()):
        to_append = {
            'head': item[1]['head'],
            'head_replaced': item[1]['head_replaced'],
            'head_translated': item[1]['head_translated'],
    }
    if px.match(item[1]['head']) == None:
        noPerson_list.append(to_append)
    else:
        withPerson_list.append(to_append)
    withPerson = pd.DataFrame(withPerson_list)
    noPerson = pd.DataFrame(noPerson_list)
    withPerson.to_csv(config["path"]["head_shortSentence.csv"], encoding="utf_8_sig")
    noPerson.to_csv(config["path"]["head_phrase.csv"], encoding="utf_8_sig")

def build_ATOMIC():
    global joint_translated_table
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    atomic = pd.read_csv(config["path"]["all_unique_notNone"], sep='\t')
    head = pd.read_csv(config["path"]["unique_head_replaced_translated"])
    tail = pd.read_csv(config["path"]["unique_tail_replaced_translated"])
    head_table = build_translated_table(head, 'head', 'head_translated')
    tail_table = build_translated_table(tail, 'tail', 'tail_translated')

    atomic_Chinese = pd.DataFrame()
    count = 0
    for idx,item in atomic.iterrows():
        head_key = item['head']
        tail_key = item['tail']
        relation = item['relation']
        # print(head_key,tail_key)
        if head_key in joint_translated_table.keys() and tail_key in joint_translated_table[head_key].keys() and relation in joint_translated_table[head_key][tail_key].keys():
            head_translated = joint_translated_table[head_key][tail_key]['head_translated']
            tail_translated = joint_translated_table[head_key][tail_key]['tail_translated']
            count += 1
        else:
            head_translated = head_table[head_key]
            tail_translated = tail_table[tail_key]
        data = {
            'head':head_translated,
            'relation': relation,
            'tail': tail_translated,
        }
        atomic_Chinese = atomic_Chinese.append(data,ignore_index=True)
        print('{}/{}'.format(idx, len(atomic)))
    print(count)
    atomic_Chinese.to_csv(config["path"]["ATOMIC_Chinese"], sep='\t')

def main():
    print('building ATOMIC_Chinese...')
    build_joint_translated_table()
    build_ATOMIC()
    filter_phrase()


if __name__ == '__main__':
    main()