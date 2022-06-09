# C3KG

## Introduction
Existing commonsense knowledge bases often organize tuples in an isolated manner, which is deficient for commonsense conversational models to plan the next steps. To fill the gap, we curate a large-scale multi-turn human-written conversation corpus, and create the first Chinese commonsense conversation knowledge graph which incorporates both social commonsense knowledge and dialog flow information. To show the potential of our graph, we develop a graph-conversation matching approach, and benchmark two graph-grounded conversational tasks. The paper "C3KG: A Chinese Commonsense Conversation Knowledge Graph" has been accepted by Findings of 60th Annual Meeting of the Association for Computational Linguistics(Findings of ACL 2022). For details, https://aclanthology.org/2022.findings-acl.107/

## Resource Released
We put all of our released resource [here](https://drive.google.com/drive/folders/1ScEEbRjpUgc2JxmQ6ITwTwIQXIKSUjlI?usp=sharing), including __C3KG__,  __ATOMIC_ZH__ and __CConv__ dataset

## Quick Start

### Data and Models Preparation

* Download [ATOMIC2020](https://allenai.org/data/atomic-2020) dataset and put all of three data files(__train.tsv__, __test.tsv__, __dev.tsv__) into __./data__:
```bash
wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
unzip atomic2020_data-feb2021.zip
cd atomic2020_data-feb2021
cp train.tsv ../data/
cp test.tsv ../data/
cp dev.tsv ../data/
```

* Download [LTP4](https://github.com/HIT-SCIR/ltp) toolkit(here we use __Base2__ model). Create __./model__ and put the Base2 model into it.
```bash
wget http://39.96.43.154/ltp/v3/base2.tgz
tar -xzvf base2.tgz
mkdir model
mv Base2 ./model/
```

* Download our SBERT-ATOMIC semantic similarity model [here](https://drive.google.com/drive/folders/1oMDCAJGBfLkBTQTcslkwl1XhRUnmAE5w?usp=sharing) and put it into __./model__.


### Data Preprocess
* Rewrite the __request_dev()__ function in __./preprocess/get_trans.py__ using any translation model or API:
```python
def request_dev(query):
    # rewrite using any translation model or API
    raise NotImplementedError("rewrite using any translation model or API")
```
* After that, run __preprocess.sh__:
```bash
chmod 777 preprocess.sh
./preprocess.sh
```
* Or you can use the translated __ATOMIC_Chinese.tsv__, __head_shortSentence.csv__,__head_phrase.csv__ [here](https://drive.google.com/drive/folders/1SsQqnUgUktyx_df-dzphXcs72HYI5cje?usp=sharing) directly.


### C3KG Construction
* To get C3KG, run __construct.sh__, note that we put the __CConv__ dataset [here](https://drive.google.com/drive/folders/1SsQqnUgUktyx_df-dzphXcs72HYI5cje?usp=sharing):
```bash
chmod 777 construct.sh
./construct.sh
```

## Licence
* Our dataset is licensed under the CC BY 4.0 and our code is licensed under the Apache License 2.0.