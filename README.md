# Continual Training of Language Models for Few-Shot Learning

This repository contains the code and pre-trained models for our EMNLP'22 paper Continual Training of Language Models for Few-Shot Learning by Zixuan Ke, Haowei Lin, Yijia Shao, Hu Xu, Lei Shu, and Bing Liu.


## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
  - [Use CPT with Huggingface](#use-cpt-with-huggingface)
  - [Train CPT](#train-cpt)
    - [Requirements](#requirements)
    - [Data](#data)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

We propose the problem of continually extending an LM by incrementally post-train the LM with a sequence of unlabeled domain corpora to expand its knowledge without forgetting its previous skills. Under the goal of improving few-shot end-task learning in these domains, we propose a system called CPT (Continual Post-Training), which to our knowledge, is the first continual post-training system. Experimental results verify its effectiveness. And the following figure is an illustration of our model.

![](figures/model.png)

## Use CPT with Huggingface

You can easily import our continually post-trained model with HuggingFace's `transformers`:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import our model. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("UIC-Liu-Lab/CPT", trust_remote_code=True)

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Task id and smax
t = torch.LongTensor([0]).to(model.device)	# using task 0's CL-plugin, choose from {0, 1, 2, 3}
smax = 400

# Get the model output!
res = model(**inputs, return_dict=True, t=t, s=smax)
```

If you encounter any problem when directly loading the models by HuggingFace's API, you can also download the models manually from the [repo](https://huggingface.co/UIC-Liu-Lab/CPT/tree/main) and use `model = AutoModel.from_pretrained({PATH TO THE DOWNLOAD MODEL})`.

Note: The post-trained weights you load contain un-trained classification heads. The post-training sequence is `Restaurant -> AI -> ACL -> AGNews`, you can use the downloaded weights to fine-tune the corresponding end-task. The results (MF1/Acc) will be consistent with follows.

|                 | Restaurant    | AI            | ACL           | AGNews        | Avg.          |
| --------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| UIC-Liu-Lab/CPT | 53.90 / 75.13 | 30.42 / 30.89 | 37.56 / 38.53 | 63.77 / 65.79 | 46.41 / 52.59 |

## Train CPT

In the following section, we describe how to train a CPT model by using our code.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.0` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.0` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.0
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Data

Before training and evaluation, please download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/1-8Ly4Jk12LHnkMKnxhuXieVNMA9cyORQ?usp=sharing) and save them in the `./data` directory. 

### Post-Training

**Training scripts**

We provide an example training script to run CPT. We explain the arguments in the following:

* `--pt_task`: The id for the post-train task. e.g. `--pt_task 3` means post-train the model on the fourth dataset. 
* `--idrandom`: choose the task sequence. See `./sequence` for more details.
  * You can post-train CPT using other task sequences by modifying this argument.
* `--ffn_adapter_size`: The size of the adapters that are plugged into the FFN layer. See our paper for details.
* `--attn_adapter_size`: The size of the adapters that are plugged into the attention layer. See our paper for details.
* `--baseline`: The name of the model. Our codebase only supports `cpt_parallel` and `cpt_sequential` currently.
  * Actually, our codebase is very flexible for adding more baselines. We will add more baselines in the future.


All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--max_seq_length`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to train and evaluate the model on the `cpt_datasets_pt` and `cpt_datasets_ft` sequence files. See `./sequence` for details.

For the results in the paper, we use Nvidia GeForce RTX2080 GPUs with CUDA 10. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

**Hyperparameters**

We use the following hyperparameters for training CPT:

|               | CPT post-training | CPT fine-tuning |
| :------------ | :---------------: | :-------------: |
| Batch size    |        48         |       20        |
| Learning rate |       1e-4        |      5e-5       |
| Epoch         |         1         |       20        |

### End-Task Fine-tuning

Once you finished post-train, come back to the root directory and simply run

```bash
CUDA_VISIBLE_DEVICES=${your_cuda_device_id} bash scripts/finetune_cpt_unfreeze_parallel.sh
```
Our codebase offers convenient tools for collecting experimental results and automatic scripts for continual learning. After the right execution, you are expected to get the results in the following format:

```

└── seq0
    ├── seed111
    │   └── cpt_parallel_unfreeze
    │       └── pt
    │           ├── acl_unsup_roberta
    │           │   ├── 111.model
    │           │   ├── config.json
    │           │   ├── mask_back
    │           │   ├── mask_pre
    │           │   ├── merges.txt
    │           │   ├── pt_log
    │           │   │   └── events.out.tfevents.1665472004.lthpc.1718401.0
    │           │   ├── pytorch_model.bin
    │           │   ├── special_tokens_map.json
    │           │   ├── tokenizer_config.json
    │           │   └── vocab.json
    │           ├── agnews_unsup_roberta
    │           │   └──  ...
    │           ├── ai_unsup_roberta
    │           │   └──  ...
    │           ├── few_shot_acc_111
    │           ├── few_shot_f1_111
    │           ├── few_shot_forward_acc_111
    │           ├── few_shot_forward_f1_111
    │           ├── few_shot_progressive_acc_111
    │           ├── few_shot_progressive_f1_111
    │           ├── restaurant_unsup_roberta
    │           └── └──  ...
    ├── seed2021
    │   └── cpt_parallel_unfreeze
    │       └── pt
    │           ├── few_shot_acc_2021
    │           ├── few_shot_f1_2021
    │           ├── few_shot_forward_acc_2021
    │           ├── few_shot_forward_f1_2021
    │           ├── few_shot_progressive_acc_2021
    │           └── few_shot_progressive_f1_2021
    └── seed222
        └── cpt_parallel_unfreeze
            └── pt
                ├── few_shot_acc_222
                ├── few_shot_f1_222
                ├── few_shot_forward_acc_222
                ├── few_shot_forward_f1_222
                ├── few_shot_progressive_acc_222
                └──  few_shot_progressive_f1_222
```

Arguments for the end-task fine-tuning script are as follows,

* `--pt_task`: The id for the post-train task. e.g. `--pt_task 3` means using the model after continually post-trained on the four datasets. 
* `ft_task`: The id for the fine-tuning task. e.g. `--ft_task 0` means doing fine-tuning on the first dataset.
* `--idrandom`: choose the task sequence. See `./sequence` for more details.
    * You can post-train CPT using other task sequences by modifying this argument.
* `--pt_seed`: the seed used for post-training, used to find the right checkpoint dir of post-trained models.
* `--unfreeze_lm`: whether to unfreeze the backbone (Roberta) when fine-tuning.
* `--ffn_adapter_size`: The size of the adapters that are plugged into the FFN layer. See our paper for details.
* `--attn_adapter_size`: The size of the adapters that are plugged into the attention layer. See our paper for details.

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email [Zixuan](`zke4@uic.edu`), [Haowei](`linhaowei@pku.edu.cn`), and [Yijia](shaoyj.pku.edu.cn). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Our paper will be released soon!
