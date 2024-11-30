#!/usr/bin/env python
# coding: utf-8

# In[10]:


from datasets import load_dataset
from setfit import SetFitModel
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()


# In[11]:


saved_model = SetFitModel._from_pretrained('./classify/setfit/model')


# In[12]:


# Run model on headlines!
df = pd.read_csv("./data/headlines.csv").dropna(subset=['Headline'])
preds = saved_model.predict(df['Headline']) # on cpu :(


# In[ ]:


# Save predictions to a csv file
preds = pd.DataFrame({'BreachMentioned': [bool(p[0]) for p in preds], 'CompanyMenioned': [bool(p[1]) for p in preds]})
df = pd.concat([df, preds], axis=1)
df


# In[ ]:


df.to_csv("./data/setfit_predictions.csv", index=False)

