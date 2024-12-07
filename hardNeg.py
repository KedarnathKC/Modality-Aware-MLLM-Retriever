from datasets import load_dataset, concatenate_datasets, Features, Value, load_from_disk, Dataset, logging 
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from PIL import Image
import requests
import json
from huggingface_hub import login
import pandas as pd
import random
import copy
from tqdm import tqdm


os.environ['TRANSFORMERS_CACHE'] = '/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/model'
hf_token = os.getenv("hf_token")
login(token=hf_token)

data_files_train = {
        'train': ['query/train/' + f'*{file}*' + '.jsonl' for file in ['visualnews','fashion200k','mscoco']]
    }

ds_train = load_dataset("TIGER-Lab/M-BEIR",
                        cache_dir='dataset/train',
                        data_files=data_files_train,
                        name='query', split=f'train')

path = '/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/dataset/cands/'
datasets = {}
dataset_ids=[0,1,9]
for id in dataset_ids:
    load_path = f'{id}'
    datasets[id] = load_from_disk(path+load_path)

# Read the TSV file into a DataFrame
instructions = pd.read_csv('/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/datasetUtils/query_instructions.tsv', sep='\t')
instructions.head()

dataset_id = 0
query_modality ='text'

path = '/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/dataset/trains/'
load_path = f'{dataset_id}_{query_modality}'
dataset = load_from_disk(path + load_path) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained('nvidia/MM-Embed', trust_remote_code=True, cache_dir = '/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/model').half().to(device)

# Disable progress bar
logging.set_verbosity_error()
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "True"

passage_mods = {'text':'txt', 'image':'img'}
passsage_ret = {'text':'txt', 'image':'img_path'}
mods = {'text':'image', 'image':'text'}
hard_negs={}
pos_in_range={}
# for i in range(len(ds_train)):
cand_size = 300
query_batch_size = 32
max_length = 512
batch_size=100
    
for i in tqdm(range(0,len(dataset),query_batch_size), desc="Processing Items", leave=True):
    instr = instructions[(instructions['query_modality'] == query_modality) & (instructions['cand_modality'] == mods[query_modality]) & (instructions['dataset_id'] == dataset_id)]['prompt_'+str(random.randint(1,4))].tolist()[0]
    passages=[]
    
    candidate_pool = copy.deepcopy(datasets[dataset_id]['did'])
    
    instr = instructions[(instructions['query_modality'] == query_modality) & (instructions['cand_modality'] == mods[query_modality]) & (instructions['dataset_id'] == dataset_id)]['prompt_'+str(random.randint(1,4))].tolist()[0]
    inputs=[]
    for j in range(i,i+query_batch_size):
        inp={'txt':dataset[j]['query_txt']}
        if query_modality == 'image':
            inp['img']=Image.open('dataset/'+dataset[j]['query_img_path'])
        inputs.append(inp)
        candidate_pool.remove(dataset[j]['pos_cand_list'][0])
        
    candidate_pool = random.sample(candidate_pool,cand_size-query_batch_size)
    for j in range(i,i+query_batch_size):
        candidate_pool.append(dataset[j]['pos_cand_list'][0])
    candidate_pool = list(set(candidate_pool))
    fil_dat = list(datasets[dataset_id].filter(lambda example: example['did'] in candidate_pool, num_proc = 8))
    
    query_embeddings = model.encode(inputs, is_query=True, instruction=instr, max_length=max_length)['hidden_states']
    
    all_scores=[]
    for i in range(0,len(fil_dat),batch_size):
        passages=[]
        for qry in fil_dat[i:i+batch_size]:
            if qry['modality']=='image':
                passages.append({passage_mods[qry['modality']] : Image.open('dataset/'+qry[passsage_ret[qry['modality']]])})
            else:
                passages.append({passage_mods[qry['modality']] : qry[passsage_ret[qry['modality']]]})
        passage_embeddings = model.encode(passages, max_length=max_length)['hidden_states']
        score = (query_embeddings @ passage_embeddings.T) * 100
        all_scores.append(score.cpu())
    scores = torch.cat(all_scores, dim=1)
    top_scores, top_indices = torch.topk(scores, 50, largest=True, sorted=True)

    for j in range(query_batch_size):
        # Check if the positive candidate is in the top indices
        if cand_size - query_batch_size + j in top_indices[j]:
            pos_in_range[dataset[i + j]['qid']] = 1
            index = torch.where(top_indices[j] == cand_size - query_batch_size + j)[0].item()
        else:
            pos_in_range[dataset[i + j]['qid']] = 0
            index = 50

        # Get the top candidates (C1) and harder negatives (C2) efficiently
        top_indice = top_indices[j].tolist()

        # Separate candidates into C1 and C2
        C1_dids = {candidate_pool[k] for k in top_indice[:index]}
        C2_dids = {candidate_pool[k] for k in top_indice[39:]}

        # Filter once to find all relevant candidates
        C1_and_C2_dids = C1_dids | C2_dids
        relevant_fil_dat = [qry for qry in fil_dat if qry['did'] in C1_and_C2_dids]

        # Split filtered data into C1 and C2
        C1 = [qry['did'] for qry in relevant_fil_dat if qry['did'] in C1_dids and qry['modality'] == query_modality]
        C2 = [qry['did'] for qry in relevant_fil_dat if qry['did'] in C2_dids and qry['modality'] != query_modality]

        # Combine results
        C1.extend(C2)
        
        # Update hard negatives
        hard_negs[dataset[i + j]['qid']] = C1

# Save results
with open(f"/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/hardNeg_{dataset_id}_{query_modality}.json", "w") as json_file:
    json.dump(hard_negs, json_file, indent=4)

with open(f"/work/pi_pgrabowicz_umass_edu/kchimmad/cs646/Modality-Aware-MMLM-Retriever/posInRange_{dataset_id}_{query_modality}.json", "w") as json_file:
    json.dump(pos_in_range, json_file, indent=4)


    
