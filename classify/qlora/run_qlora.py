from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
import pandas as pd
import torch
import time
import os

TO_LABEL_PATH        = './testset.csv' #'/gscratch/stf/lleibm/data/headlines.csv'
OUT_PATH             = '/gscratch/stf/lleibm/data/qlora_gpt_classified.csv'
PREFIX               = open('./llm_classify/prompt_QLora.txt', 'r').read()


TO_LABEL_PATH        = './llm_classify/testset.csv' #'/gscratch/stf/lleibm/data/headlines.csv'


schema = {
    'BreachMentioned': 'true|True|false|False',
    'CompanyMentioned': '\w{3,}', 
}

MODEL_DIR           = '/gscratch/stf/lleibm/decilm-7b-headline-qlora'
generation_kwargs = {
    "max_new_tokens": 50,
    "early_stopping": True,
    "num_beams": 5,
    "temperature" : 0.001,
    "do_sample":True,
    # "no_repeat_ngram_size": 3,
    # "repetition_penalty" : 1.5,
    # "renormalize_logits": True
}

BASE_MODEL =  "Deci/DeciLM-7B"
AutoTokenizer.from_pretrained = partial(AutoTokenizer.from_pretrained, trust_remote_code=True)

instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype = torch.bfloat16,
    device_map  = 'auto',
    trust_remote_code=True,
)

merged_model = instruction_tuned_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from transformers import pipeline

decilm_tuned_pipeline = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    **generation_kwargs
)

def query_batch(prompt_texts:list[str]) -> dict[str: str]:
    return {p: x[0]['generated_text'] for p, x in zip(prompt_texts, decilm_tuned_pipeline(prompt_texts))}

def to_prompt(row:pd.Series, index=True, columns=True, index_names=None) -> str:
    val = list()
    if index:
        if not index_names:
            index_names = row.index.names
        if not index_names:
            index_names = ['index']
        if len(index_names) != len(row.name):
            raise Exception("Index names must be the same length as the index")
        for i, name in enumerate(index_names):
            val.append(f"{name}: {row.name[i]}")
    
    if columns:
        for col, value in row.items():
            val.append(f"{col} : {value}")
    return '\n'.join(val)

def construct_fragment(sample, prefix=PREFIX, include_solution=False) -> str:
  prompt = "<s>"
  if prefix: prompt += prefix
  prompt += to_prompt(sample, index=True, columns=False, index_names=['Date', 'Publication', 'Headline', 'URL'])
  prompt += " ###> "
  if include_solution:
      prompt += to_prompt(sample, index=False, columns=True)
      prompt += "</s>"
  return prompt

def fix_set(frame):
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')] # Drop all Unnamed: columns
    frame['Headline'] = frame['Headline'].apply(str.strip)
    frame.drop_duplicates(subset=['Headline'], inplace=True)
    frame.drop_duplicates(subset=['URL'], inplace=True)

    frame.set_index(['Date', 'Publication', 'Headline', 'URL'], inplace=True)
    frame.fillna(False, inplace=True)
    return frame

to_label = fix_set(pd.read_csv(TO_LABEL_PATH))

if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
to_label[:0].to_csv(OUT_PATH)

start_time = time.monotonic()

from tools import chunkify
from copy import copy
import numpy as np
import traceback

for chunk in chunkify(to_label, 50, use_tqdm=True):    
    # Calculate prompts
    prompts = {idx: construct_fragment(row) for idx, row in chunk.iterrows()}
    
    # Chunked query
    responses = query_batch([prompt_str for prompt_str in prompts.values()])

    # Parse each response
    new_rows = pd.DataFrame()
    for idx, row in chunk.iterrows():
        try:
            parsed_response = dict()
            response = responses.get(prompts.get(idx, None), "")
            
            # Delete everything before ##>
            if '###>' in response:
                response = response[response.find('###>')+4:]
            
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
        
        new_rows = pd.concat([new_rows, row_copy])
        
    # Save all new rows
    if len(new_rows) > 0:
        new_rows.reset_index().to_csv(OUT_PATH, mode='a', header=False, index=False)


elapsed_time = time.monotonic() - start_time
print(f'Processed {len(to_label):,d} records in {elapsed_time/60:.1f} minutes.')
