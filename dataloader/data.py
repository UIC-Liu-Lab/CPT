import json
import os.path
import random

import jsonlines
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset

implemented_datasets = [
    'restaurant_unsup',
    'ai_unsup',
    'acl_unsup',
    'agnews_unsup',
    'restaurant_sup',
    'scierc_sup',
    'aclarc_sup',
    'agnews_sup'
]

dataset_class_num = {
    'restaurant_sup': 3,
    'agnews_sup': 4,
    'aclarc_sup': 6,
    'scierc_sup': 7,
}


def get_restaurant_unsup():
    data_files = {'train': './data/restaurant_unsup/yelp_restaurant.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets


def get_agnews_unsup():
    data_files = {'train': './data/agnews_unsup/ag_news_corpus.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets


def get_acl_unsup():
    data_files = {'train': './data/acl_unsup/acl_anthology.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets


def get_ai_unsup():
    data_files = {'train': './data/ai_unsup/ai_corpus.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets


def get_restaurant_sup(tokenizer):
    root_dir = './data/restaurant_sup'

    def label2idx(label):
        if label == 'positive':
            return 0
        elif label == 'neutral':
            return 1
        else:
            return 2

    new_data = {}
    for ds in ['train', 'test']:
        new_data[ds] = {}
        new_data[ds]['text'] = []
        new_data[ds]['labels'] = []
        with open(os.path.join(root_dir, ds + '.json')) as f:
            data = json.load(f)
        for _data in data:
            new_data[ds]['text'].append(
                data[_data]['term'] + ' ' + tokenizer.sep_token + data[_data]['sentence'])
            new_data[ds]['labels'].append(label2idx(data[_data]['polarity']))
    datasets = DatasetDict(
        {
            'train': Dataset.from_dict(new_data['train']),
            'test': Dataset.from_dict(new_data['test'])
        }
    )

    return datasets


def get_dataset(dataset_name, tokenizer, args):
    # --- Unsupervised Learning datasets ---
    # attributes: 'text'

    if dataset_name == 'restaurant_unsup':
        datasets = get_restaurant_unsup()

    elif dataset_name == 'agnews_unsup':
        datasets = get_agnews_unsup()

    elif dataset_name == 'acl_unsup':
        datasets = get_acl_unsup()

    elif dataset_name == 'ai_unsup':
        datasets = get_ai_unsup()

    # --- Supervised Learning datasets ---
    # attributes: 'text', 'labels'

    elif dataset_name == 'restaurant_sup':
        datasets = get_restaurant_sup(tokenizer)

    elif dataset_name == 'agnews_sup':
        df = pd.read_csv('./data/agnews_sup/agnews_sup.csv',
                         names=['label', 'title', 'description'])

        datasets = Dataset.from_pandas(df)

        def combine_function(example):
            example['text'] = example['description']
            example['labels'] = example['label'] - 1
            return example

        datasets = datasets.map(combine_function,
                                batched=False,
                                num_proc=16,
                                remove_columns=['label', 'title', 'description'])

        datasets = datasets.train_test_split(
            test_size=0.1, seed=2021, shuffle=True)

    elif dataset_name == 'aclarc_sup':
        label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2,
                     'Motivation': 3, 'Extends': 4, 'Background': 5}
        new_data = {}
        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/acl_sup/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])

        # re-partition by classes to make it balanced.
        train_ratio = 0.9
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])

        for label in range(num_label):
            num_takeout = int((total_num * (1 - train_ratio)) // num_label)
            label_pos = [lab_id for lab_id, lab in enumerate(
                new_data['train']['labels']) if lab == label][:num_takeout]
            label_takeout = [lab for lab_id, lab in enumerate(
                new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id, lab in enumerate(
                new_data['train']['text']) if lab_id in label_pos]

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id, lab in enumerate(new_data['train']['labels']) if
                                           lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id, lab in enumerate(new_data['train']['text']) if
                                         lab_id not in label_pos]

        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    elif dataset_name == 'scierc_sup':
        label2idx = {'FEATURE-OF': 0, 'CONJUNCTION': 1, 'EVALUATE-FOR': 2, 'HYPONYM-OF': 3, 'USED-FOR': 4,
                     'PART-OF': 5, 'COMPARE': 6}
        new_data = {}
        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/scierc_sup/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])

        # re-partition by classes to make it balanced.
        train_ratio = 0.7
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ",
              len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1 - train_ratio)) // num_label)
            label_pos = [lab_id for lab_id, lab in enumerate(
                new_data['train']['labels']) if lab == label][:num_takeout]
            label_takeout = [lab for lab_id, lab in enumerate(
                new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id, lab in enumerate(
                new_data['train']['text']) if lab_id in label_pos]

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id, lab in enumerate(new_data['train']['labels']) if
                                           lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id, lab in enumerate(new_data['train']['text']) if
                                         lab_id not in label_pos]

        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    if 'few_shot' in args and args.few_shot:
        seed = 2022
        num_labels = max(datasets['train']['labels'])
        assert dataset_class_num[dataset_name] == num_labels + 1
        ## 32 samples for tasks less than 5 labels, 
        ## 8 samples per class for task with 5 or more than 5 labels
        if num_labels < 4:
            datasets['train'] = datasets['train'].shuffle(seed=seed)
            datasets['train'] = datasets['train'].select(range(32))
        else:
            datasets['train'] = datasets['train'].shuffle(seed=seed)
            _idx = [[] for i in range(num_labels+1)]
            for idx, label in enumerate(datasets['train']['labels']):
                if len(_idx[label]) < 8:
                    _idx[label].append(idx)
            idx_lst = [i for item in _idx for i in item]
            datasets['train'] = datasets['train'].select(idx_lst).shuffle(seed=seed)

    return datasets
