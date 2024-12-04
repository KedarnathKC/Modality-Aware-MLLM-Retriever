from accelerate.utils import MODEL_NAME
from transformers import CLIPModel, CLIPProcessor
from datasets import concatenate_datasets
from datasetUtils.load_dataset import get_validation_data, get_candidate_dataset
import os
import torch
import random
from PIL import Image
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')

MODEL_NAME = 'openai/clip-vit-base-patch16'
dataset_path = '/Users/ykamoji/Documents/ImageDatabase/m-beir/mbeir_images_train'

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir='model/')

def load_query_instruction():
    prompts_dict = {}
    with open('datasetUtils/query_instructions.tsv', "r") as f:
        next(f)  # Skip the header line
        for line in f.readlines():
            parts = line.strip().split("\t")
            # Construct the key to be dataset_id, query_modality, cand_modality
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
            prompts_dict[key] = prompts

    return prompts_dict

query_instructions = load_query_instruction()

cache_splits = "_".join(['visualnews', 'fashion200k', 'mscoco'])

with open(f"cache/.cache_cand_{cache_splits}.json", 'r') as f:
    candidate_dataset = json.load(f)

queries = get_validation_data(domains=['mscoco','visualnews','fashion200k'])

get_candidate_dataset = get_candidate_dataset(domains=['mscoco', 'visualnews', 'fashion200k'])

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s


def get_random_query_prompt(prompts_dict, dataset_id, query_modality, cand_modality):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = prompts_dict.get(key, [])
    assert prompts, f"Cannot find prompts for {key}"
    prompt = format_string(random.choice(prompts))
    assert prompt, f"Prompt is empty for {key}"
    return prompt


def _get_padded_text_with_mask(txt):
    return (txt, 1) if txt not in [None, ""] else ("", 0)

def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (torch.zeros((3, 224, 224)), 0)

def load_and_preprocess_image(query_img_path):
    """Load an image given a path"""
    if not query_img_path:
        return None
    image = Image.open(dataset_path + '/' + query_img_path).convert("RGB")
    image = feature_processor(images=image, return_tensors='pt')["pixel_values"].squeeze(0)
    return image


def prepare_data_dict(txt, img_path):
    img = load_and_preprocess_image(img_path)
    return {"txt": txt, "img": img}


def get_random_candidates(dataset_id, modality):
    candidates_1 = get_candidate_dataset.filter(lambda x: x['did'].startswith(f"{dataset_id}:") and x['modality'] == modality).shuffle().select(range(2))
    candidates_2 = get_candidate_dataset.filter(lambda x: x['did'].startswith(f"{dataset_id}:") and x['modality'] != modality).shuffle().select(range(2))
    return concatenate_datasets([candidates_1, candidates_2])


def get_ranking(queries, model):

    model.eval()
    model.to(Device)
    with torch.no_grad():
        for q in queries:
            query_dataset_id = q['qid'].split(":")[0]
            query_modality = q['query_modality']
            pos_cand_did = q['pos_cand_list'][0]
            pos_cand = candidate_dataset[pos_cand_did]
            pos_cand_modality = pos_cand['modality']
            inst_query = get_random_query_prompt(query_instructions, query_dataset_id, query_modality, pos_cand_modality)
            query_txt_with_prompt = format_string(f"{inst_query} {q['query_txt']}")

            query = prepare_data_dict(query_txt_with_prompt, q["query_img_path"])
            positive_candidate = prepare_data_dict(pos_cand['txt'], pos_cand['img_path'])
            random_cand = get_random_candidates(query_dataset_id, pos_cand_modality)
            candidates = [
                    prepare_data_dict(
                        format_string(cand["txt"]),
                        cand["img_path"]
                    )
                    for cand in random_cand
                ]

            candidates.insert(0, positive_candidate)
            instance_keys = ['query', 'candidates']

            batch = [{'query': query,'candidates': candidates}]
            index_mapping = {
                "query": [[]],
            }
            index_mapping.update({"candidates": [[]]})

            txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []
            counter = 0
            for inst_idx, instance in enumerate(batch):
                for instance_key in instance_keys:
                    items = [instance[instance_key]] if instance_key != 'candidates' else instance[instance_key]  # list
                    for item in items:
                        txt = item["txt"]
                        img = item["img"]

                        index_mapping[instance_key][inst_idx].append(counter)  # Track current index
                        counter += 1
                        padded_txt, txt_mask = _get_padded_text_with_mask(txt)
                        padded_img, img_mask = _get_padded_image_with_mask(img)
                        txt_list.append(padded_txt)
                        img_list.append(padded_img)
                        txt_mask_list.append(txt_mask)
                        img_mask_list.append(img_mask)

            tokenized_text = feature_processor(text=txt_list, padding=True, truncation=True, return_tensors='pt').to(Device)

            txt_batched = tokenized_text['input_ids']
            txt_attention_mask_batched = tokenized_text['attention_mask']

            txt_mask_list = torch.tensor(txt_mask_list, dtype=torch.long, device=Device).unsqueeze(-1)
            img_mask_list = torch.tensor(img_mask_list, dtype=torch.long, device=Device).unsqueeze(-1)

            txt_emb = model.get_text_features(txt_batched, txt_attention_mask_batched) * txt_mask_list
            img_emb = model.get_image_features(torch.stack(img_list, dim=0)) * img_mask_list
            embeddings = img_emb + txt_emb

            q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]
            cand_embeds = embeddings[torch.tensor(index_mapping["candidates"]).flatten()]

            scores = (q_embeds @ cand_embeds.transpose(0, 1)).cpu().numpy()

            print(scores)
            top_ids = np.argsort(scores)[::-1]
            print(top_ids)


def start_inference(total_examples_count):
    HF_CLIP = CLIPModel.from_pretrained(MODEL_NAME, cache_dir="model/")

    # ## Update run number here for the random trained clip model
    # RAND_CLIP = CLIPModel.from_pretrained("", cache_dir="model/")
    #
    # ## Update run number here for the hard negative trained clip model
    # HARD_NEG_CLIP = CLIPModel.from_pretrained("", cache_dir="model/")
    #
    # ## Update run number here for the model tuned after the Random model fine-tuning 4 or 3 epochs (whichever is better)
    # HARD_NEG_FT = CLIPModel.from_pretrained("", cache_dir="model/")

    task_1_queries = queries.filter(lambda x: x['query_modality'] == 'text').select(range(total_examples_count))

    task_2_queries = queries.filter(lambda x: x['query_modality'] == 'image').select(range(total_examples_count))

    get_ranking(task_1_queries, HF_CLIP)





if __name__ == '__main__':
    start_inference(3)