#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datasets import load_dataset
from setfit import SetFitModel
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()


# In[3]:


breach_model  = SetFitModel._from_pretrained('/gscratch/stf/lleibm/models/model_breach')
company_model = SetFitModel._from_pretrained('/gscratch/stf/lleibm/models/model_company')


# In[6]:


# Run model on headlines!
DATA_PATH = '/gscratch/stf/lleibm/data/headlines.csv' #"./data/headlines.csv"
df = pd.read_csv(DATA_PATH).dropna(subset=['Headline'])

pred_breach  = breach_model.predict(df['Headline'].reset_index(drop=True)) # on cpu :(
pred_company = company_model.predict(df['Headline'].reset_index(drop=True)) # on cpu :(


# In[12]:


sims  = 0
diffs = 0
b, c = 0, 0
for x, y in zip(pred_breach, pred_company):
    if x == y: sims += 1
    else:      diffs += 1
    if x and not y: b += 1
    if y and not x: c += 1

print(f"Columns labelled Breach AND Company: {sims}")
print(f"Columns labelled Breach XOR Company: {diffs}")
print(f"Columns labelled Breach and NOT Company: {b}")
print(f"Columns labelled NOT Company and Breach: {c}")


# In[7]:


# Save predictions to a csv file
preds = pd.DataFrame({'BreachMentioned': [bool(p) for p in pred_breach], 'CompanyMenioned': [bool(p) for p in pred_company]})
df = pd.concat([df, preds], axis=1)
df


# In[8]:


df.to_csv("/gscratch/stf/lleibm/data/setfit_predictions.csv", index=False)

