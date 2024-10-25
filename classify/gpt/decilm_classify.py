import os
import time
import re
import warnings
import datetime
import json

from copy import copy
from dataclasses import dataclass, field
import traceback

import numpy as np
import pandas as pd
from itertools import chain

from tqdm.auto import tqdm
tqdm.pandas()

from tools import embedding_pipeline

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

TRAINING_DATA_PATH   = './trainset.csv'
TO_LABEL_PATH        = './testset.csv' #'/gscratch/stf/lleibm/data/headlines.csv'
OUT_PATH             = '/gscratch/stf/lleibm/data/gpt_classified.csv'
NUM_EXAMPLES         = 6

MAX_TOKENS           = 40 # only include response; smaller helps with ratelimiting
TEMPERATURE          = 1e-4 # using 0 based on xinyi, down from 1.4 earlier

schema = {
    'BreachMentioned': 'true|True|false|False',
    'CompanyMentioned': '\w{3,}', 
}

# Load training set
train_set = pd.read_csv(TRAINING_DATA_PATH)
train_set = train_set.loc[:, ~train_set.columns.str.contains('^Unnamed')] # Drop all Unnamed: columns
train_set.drop_duplicates(subset=['Headline'], inplace=True)
train_set.drop_duplicates(subset=['URL'], inplace=True)

train_set['Headline'] = train_set['Headline'].apply(str.strip)
train_set.set_index(['Date', 'Publication', 'Headline', 'URL'], inplace=True)
train_set.fillna(False, inplace=True)

# Load to_label EXACTLY as above
to_label = pd.read_csv(TO_LABEL_PATH)

to_label.dropna(subset=['Headline'], inplace=True)
to_label['Headline'] = to_label['Headline'].apply(str.strip)
to_label.set_index(train_set.index.names, inplace=True)

# If we are trying to label a column that doesn't exist, create it
for col in schema.keys():
    if col not in to_label.columns:
        to_label[col] = None

# Assert that the schema of to_label and train_set match
if not set(chain(train_set.index.names, train_set.columns)) <= set(chain(to_label.index.names, to_label.columns)):
    raise ValueError(f'To Label schema does not match train set schema! \n Missing Columns: {set(chain(train_set.index.names, train_set.columns)) - set(chain(to_label.index.names, to_label.columns))}')

# Shuffle with SET seed
to_label = to_label.sample(frac=1, random_state=42)
print(f'Loaded {len(to_label)} rules to classify.')

# Construct prompt
example_prompt = PromptTemplate.from_template(
    '\n'.join(f'{c} : {{{c}}}' for c in chain(train_set.index.names, train_set.columns)),
)

examples = train_set.reset_index().astype(str).to_dict(orient='records')
example_prompt.format(**examples[5])

start_time = time.monotonic()

print('Computing embeddings for examples...', end='')
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,       # This is the list[dict] of examples available to select from.
    embedding_pipeline(use_gpu=False),
    Chroma,         # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    k=NUM_EXAMPLES  # This is the number of examples to produce.
)
print(f'done in {time.monotonic()-start_time:.1f} seconds.')

with open('./prompt.txt') as f:
    prompt_prefix = f.read()
prompt_prefix += '\n\nYour answer should follow the format given in the examples:\n'
    
prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt   = example_prompt,
    input_variables  = list(train_set.index.names),
    prefix           = prompt_prefix,
    suffix           = '\nNow, classify the following:\n' + '\n'.join(chain((f'{c} : {{{c}}}' for c in train_set.index.names))),
)

print('Prompt formatting configured.\nExample prompt:')

print(
    prompt.format(
        **examples[5]
    )
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

model_name = "Deci/DeciLM-7B"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).to(device)

def query(prompt_text:str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=TEMPERATURE)
        response = tokenizer.decode(outputs[0])
        return response

start_time = time.monotonic()

to_bool = lambda x: len(str(x)) > 0 and str(x).lower() not in ['nan','false','0']

if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
to_label[:0].to_csv(OUT_PATH)

for idx, row in tqdm(to_label.iterrows(), total=to_label.shape[0]):
    prompt_args = dict(zip(to_label.index.names, idx))
    prompt_str = prompt.format(**{k: str(v) for k, v in chain(prompt_args.items(), row.items())})
    parsed_response = dict()
    
    try:
        response = query(prompt_str)
        
        # Parse yamllike response
        for line in response.split('\n'):
            if ':' in line:
                k,v = line.split(':', 1)
                if k.strip() not in parsed_response:
                    parsed_response[k.strip()] = v.strip()
        
        # Drop all keys in parsed_response that are not in schema
        for k in list(parsed_response.keys()):
            if k not in schema.keys():
                del parsed_response[k]
        
    except Exception as e:
        print(f"Error Classifying {idx}:\n{e}")
        
        # Find line number of error
        print(next((line for line in reversed(traceback.format_exc().split('\n')) if re.search(r'line \d+', line)), 'Unknown'))
        
    # Fill all missing fields with NaN
    if not set(schema.keys()) <= set(parsed_response.keys()):
        print(f"Missing keys in response for {idx}: {set(schema.keys()) - set(parsed_response.keys())}")
        print('Filling with NaN')
        for k in schema.keys():
            if k not in parsed_response:
                parsed_response[k] = np.nan
    
    # Fill in the row with the parsed response
    row_copy = copy(row).to_frame().T
    for k,v in parsed_response.items():
        if k in row_copy.columns:
            row_copy[k] = v
    
    # Save row
    row_copy.reset_index().to_csv(OUT_PATH, mode='a', header=False, index=False)
    
elapsed_time = time.monotonic() - start_time
print(f'Processed {len(to_label):,d} records in {elapsed_time/60:.1f} minutes.')