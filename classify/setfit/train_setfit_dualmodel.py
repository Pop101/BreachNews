#!/usr/bin/env python
# coding: utf-8

# In[58]:


from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, Trainer, TrainingArguments
import evaluate
import numpy as np


# In[59]:


import os
print(os.getcwd())


# In[60]:


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


# In[85]:


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

# Split into BreachMentioned and CompanyMentioned
trainset_breach = trainset.remove_columns(['CompanyMentioned'])
testset_breach  = testset.remove_columns(['CompanyMentioned'])

trainset_company = trainset.remove_columns(['BreachMentioned'])
testset_company  = testset.remove_columns(['BreachMentioned'])


# In[62]:


model_breach  = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model_company = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")


# In[63]:


trainset_breach[100]


# In[64]:
args = TrainingArguments(
    batch_size = (16, 32),
    num_epochs = (1, 16),
    output_dir = '/tmp/lleibm/checkpoints',
)

# Train setfit
trainer = Trainer(
    model=model_breach,
    args=args,
    train_dataset=trainset_breach,
    eval_dataset=testset_breach,
    column_mapping={"Headline": "text", "BreachMentioned": "label"},
)

trainer.train()

os.makedirs('/gscratch/stf/lleibm/models/model_breach', exist_ok=True)
model_breach._save_pretrained('/gscratch/stf/lleibm/models/model_breach')

metrics = trainer.evaluate()
print(metrics)


# In[86]:


# Assert that we have both 1 and 0 labels in both
assert 1 in testset_company['CompanyMentioned']


# In[87]:


# Train CompanyMentioned model
# Train setfit
trainer = Trainer(
    model=model_company,
    args=args,
    train_dataset=trainset_company,
    eval_dataset=testset_company,
    column_mapping={"Headline": "text", "CompanyMentioned": "label"},
)

trainer.train()

os.makedirs('/gscratch/stf/lleibm/models/model_company', exist_ok=True)
model_breach._save_pretrained('/gscratch/stf/lleibm/models/model_company')

metrics = trainer.evaluate()
print(metrics)


# In[ ]:


# saved_model = SetFitModel._from_pretrained('/gscratch/stf/lleibm/models/model')

