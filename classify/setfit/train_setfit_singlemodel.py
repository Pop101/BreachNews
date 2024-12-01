#!/usr/bin/env python
# coding: utf-8

# In[38]:


from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, Trainer, TrainingArguments

import evaluate
import numpy as np


# In[39]:


import os
print(os.getcwd())


# In[40]:


def to_bool(string:str) -> bool:
    string = str(string).strip().casefold()
    if string == 'nan':
        return False
    if string in ('true', 'yes', '1'):
        return True
    if string in ('false', 'no', '0', 'none'):
        return False
    return False

def dsmap(ds, col, fn):
    # I hate huggingface datasets
    return ds.map(lambda x: {**x, **{col: fn(x[col])}})


# In[46]:


# Load trainset from ../irr/train_set_gpt/caitlyn.csv
trainset = load_dataset("csv", data_files="./llm_classify/trainset.csv")["train"]
testset  = load_dataset("csv", data_files="./llm_classify/testset.csv")["train"]

# Set all columns to bool
trainset = dsmap(trainset, 'BreachMentioned', lambda x: int(to_bool(x)))
trainset = dsmap(trainset, 'CompanyMentioned', lambda x: int(bool(x)))
testset  = dsmap(testset, 'BreachMentioned', lambda x: int(to_bool(x)))
testset  = dsmap(testset, 'CompanyMentioned', lambda x: int(bool(x)))

# Remove non-input, non-target columns
trainset = trainset.remove_columns(['Date', 'Publication', 'URL'] + [x for x in trainset.column_names if x.startswith('Unnamed: ')])
testset  = testset.remove_columns(['Date', 'Publication', 'URL'] + [x for x in testset.column_names if x.startswith('Unnamed: ')])

# Remove mystery unnamed column
trainset = trainset.remove_columns([r for r in trainset.column_names if 'Unnamed' in r])
testset  = testset.remove_columns([r for r in testset.column_names if 'Unnamed' in r])


# In[47]:


# Sample rows
trainset.shuffle(seed=42)[0]


# In[ ]:


args = TrainingArguments(
    batch_size = (16, 32),
    num_epochs = (1, 16),
    output_dir = '/tmp/lleibm/checkpoints12312',
)

# f1/accuracy sentence level
multilabel_f1_metric = evaluate.load("f1", "multilabel")
multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")
def compute_metrics(y_pred, y_test):
    return {
        "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
        "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
    }


# In[48]:


model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", multi_target_strategy="multi-output")


# In[49]:


# Train!
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainset,
    eval_dataset=testset,
    column_mapping={"Headline": "text", "Target": "label"},
)


# In[50]:


trainer.train()
metrics = trainer.evaluate()
print(metrics)


# In[51]:


os.makedirs('/gscratch/stf/lleibm/models/model_dual', exist_ok=True)
model._save_pretrained('/gscratch/stf/lleibm/models/model_dual')
# saved_model = SetFitModel._from_pretrained('./classify/setfit/model')

