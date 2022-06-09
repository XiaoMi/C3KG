import pandas as pd
import re
import configparser


def merge():
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    train = pd.read_csv(config["path"]["train"], sep='\t', names=['head','relation','tail'])
    test = pd.read_csv(config["path"]["test"], sep='\t', names=['head','relation','tail'])
    dev = pd.read_csv(config["path"]["dev"], sep='\t', names=['head','relation','tail'])
    all = train.append([test, dev], ignore_index=True)
    all.to_csv(config["path"]["all"], sep='\t')

def clean():
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    df = pd.read_csv(config["path"]["all"], sep='\t', usecols=['head','relation','tail'])
    # 去重复行
    df.drop_duplicates()
    # 去null
    df = df.dropna(axis=0, how='any')
    df.to_csv(config["path"]["all_unique_notNone"], sep='\t')
    tail_unique = pd.DataFrame(data = list(set([tail for tail in df['tail']])), columns = ['tail'])
    tail_unique.to_csv(config["path"]["unique_tail"], sep='\t')
    head_unique = pd.DataFrame(data = list(set([tail for tail in df['head']])), columns = ['head'])
    head_unique.to_csv(config["path"]["unique_head"], sep='\t')
    
def replace_tail(tail):
    # 先替换person...person's（分为两个person一样和不一样两种情况）
    xxs_tail = re.compile(r'.*(Person.) .* (Person.)\'s .*',re.I)
    # 替换person...person（分为两个person一样和不一样两种情况）
    xy_tail = re.compile(r'.*(Person.) .* (Person.).*',re.I)
    px_tail = re.compile(r'.*(Person.).*',re.I)
    a_underline = re.compile(r'.* a ___.*')
    the_underline = re.compile(r'.* the ___.*')
    some_underline = re.compile(r'.* some ___.*')
    underline = re.compile(r'.* ___.*')

    #替换PersonX/PersonY
    #'...PersonX...PersonY...' -> '...PersonX...someone else...'
    #'...PersonX...PersonX's...' -> '...someone...his...'
    #'...PersonX...PersonX...' -> '...someone...himself...'
    #'...PersonX/PersonY/personx/persony...' -> 'someone'
    
    has_xxs = xxs_tail.match(tail)
    if has_xxs:
        #如果两个person一样，用someone...his替代
        if has_xxs.group(1).lower() == has_xxs.group(2).lower():
            #第一个替换成someone
            tail = re.sub(has_xxs.group(1), 'someone', tail, 1)
            #第二个替换成himself
            tail = re.sub(has_xxs.group(2)+'\'s', 'his', tail, 1)
        #如果两个person不一样，用someone...someone else's替换 
        else:
            tail = re.sub(has_xxs.group(1), 'someone', tail, 1)
            tail = re.sub(has_xxs.group(2)+'\'s', 'someone else\'s', tail, 1)
        
    has_xy = xy_tail.match(tail)
    if has_xy:
        if has_xy.group(1).lower() == has_xy.group(2).lower():
            #第一个替换成someone
            tail = re.sub(has_xy.group(1), 'someone', tail, 1)
            #第二个替换成himself
            tail = re.sub(has_xy.group(2), 'himself', tail, 1)
        else:
            tail = re.sub(has_xy.group(1), 'someone', tail, 1)
            tail = re.sub(has_xy.group(2), 'someone else', tail, 1)
        
    has_px = px_tail.match(tail)
    if has_px:
        tail = tail.replace(has_px.group(1), 'someone')
    
    #替换 ___
    #a ___ ->something
    # the ___ ->something
    # some ___ -> something
    #其他 -> something
    
    has_a_underline, has_the_underline, has_some_underline = a_underline.match(tail), the_underline.match(tail), some_underline.match(tail)
    if has_a_underline or has_the_underline or has_some_underline:
        tail = tail.replace('a ___','something').replace('the ___','something').replace('every ___','something')
    
    has_underline = underline.match(tail)
    if has_underline:
        tail = tail.replace('___', 'something')
        
    return tail

def replace_head(head):
    xy = re.compile(r'.*PersonX .* PersonY.*')
    xx = re.compile(r'.*PersonX .* PersonX.*')
    xxs = re.compile(r'.*PersonX .* PersonX\'s .*')
    px1 = re.compile(r'.*PersonX.*')
    px2 = re.compile(r'.*personx.*')
    py1 = re.compile(r'.*PersonY.*')
    py2 = re.compile(r'.*persony.*')
    a_underline = re.compile(r'.* a ___.*')
    the_underline = re.compile(r'.* the ___.*')
    some_underline = re.compile(r'.* some ___.*')
    underline = re.compile(r'.* ___.*')
    #替换PersonX/PersonY
    #'...PersonX...PersonY...' -> '...PersonX...someone else...'
    #'...PersonX...PersonX's...' -> '...someone...his...'
    #'...PersonX...PersonX...' -> '...someone...himself...'
    #'...PersonX/PersonY/personx/persony...' -> 'someone'
    
    has_xy = xy.match(head)
    if has_xy:
        head = head.replace('PersonY', 'someone else')
        
    has_xxs = xxs.match(head)
    if has_xxs:
        head = head.replace('PersonX\'s', 'his')
        head = head.replace('PersonX', 'someone')
    
    has_xx = xx.match(head)
    if has_xx:
        #第一个替换成someone
        head = re.sub('PersonX', 'someone', head, 1)
        #第二个替换成himself
        head = re.sub('PersonX', 'himself', head, 1)
        
    has_px1, has_px2, has_py1, has_py2 = px1.match(head), px2.match(head), py1.match(head), py2.match(head)
    if has_px1 or has_px2 or has_py1 or has_py2:
        head = head.replace('PersonX', 'someone').replace('personx', 'someone').replace('PersonY', 'someone').replace('persony', 'someone')
    
    #替换 ___
    #a ___ ->something
    # the ___ ->something
    # some ___ -> something
    #其他 -> something
    
    has_a_underline, has_the_underline, has_some_underline = a_underline.match(head), the_underline.match(head), some_underline.match(head)
    if has_a_underline or has_the_underline or has_some_underline:
        head = head.replace('a ___','something').replace('the ___','something').replace('every ___','something')
    
    has_underline = underline.match(head)
    if has_underline:
        head = head.replace('___', 'something')
        
    return head

def replace():
    config = configparser.ConfigParser()
    config.read("./paths.cfg")
    head = pd.read_csv(config["path"]["unique_head"], sep='\t', usecols=['head'])
    head['head_replaced'] = head['head'].apply(replace_head)
    tail = pd.read_csv(config["path"]["unique_tail"], sep='\t', usecols=['tail'])
    tail['tail_replaced'] = tail['tail'].apply(replace_tail)
    head.to_csv(config["path"]["unique_head_replaced"], sep='\t')
    tail.to_csv(config["path"]["unique_tail_replaced"], sep='\t')

def main():
    print('Pattern replacing...')
    merge()
    clean()
    replace()

if __name__ == '__main__':
    main()