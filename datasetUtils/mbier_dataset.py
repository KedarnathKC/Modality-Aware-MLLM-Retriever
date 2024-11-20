import os
import json
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from datasetUtils.load_dataset import get_training_data, get_candidate_dataset, get_validation_data

DATASET_CAN_NUM_UPPER_BOUND = 10000000  # Maximum number of candidates per dataset
DATASET_QUERY_NUM_UPPER_BOUND = 500000  # Maximum number of queries per dataset

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s


def parse_modality(modality):
    return 1 if modality == "image" else 0


def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id


def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id


def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


class MBEIRMainDataset(Dataset):
    def __init__(self, config, is_train=True, onlyPrediction=False):
        super(MBEIRMainDataset, self).__init__()

        prompts_dict = {}
        with open(config.Common.DataSet.QueryInstructionsPath, "r") as f:
            next(f)  # Skip the header line
            for line in f.readlines():
                parts = line.strip().split("\t")
                # Construct the key to be dataset_id, query_modality, cand_modality
                key = f"{parts[3]}, {parts[0]}, {parts[1]}"
                prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
                prompts_dict[key] = prompts

        self.query_instructions = prompts_dict
        self.config = config
        self.random_config = self.config.FineTuning.NegativeCandidates

        Model = config.FineTuning.Model
        DataSet = config.Common.DataSet
        self.is_train = is_train
        self.onlyPrediction = onlyPrediction
        self.feature_processor = CLIPProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)
        if is_train:
            self.ds_train, ds_candidate = get_training_data(DataSet.Train), get_candidate_dataset(DataSet.Candidate)
        else:
            self.ds_validate, ds_candidate = get_validation_data(DataSet.Test), get_candidate_dataset(DataSet.Candidate)

        self._load_cand_pool_as_dict(ds_candidate)

    def _load_cand_pool_as_dict(self, ds_candidate):
        cand_pool_dict = {}
        candidate_det = {f'{idx}': (0, 1e5, -1e5) for idx in range(10)}

        if os.path.exists(f".cache_cand_dict.json"):

            with open(f'.cache_cand_dict.json', 'r') as f:
                cand_pool_dict = json.load(f)

            with open(f'.cache_cand_count.json', 'r') as f:
                candidate_det = json.load(f)

        else:
            for cand_pool_entry in tqdm(ds_candidate, desc="Loading cand pool"):
                did = cand_pool_entry.get("did")
                cand_pool_dict[did] = cand_pool_entry
                det = did.split(':')
                dataset_id = det[0]
                doc_id = int(det[1])
                candidate_det[dataset_id] = (candidate_det[dataset_id][0] + 1,
                                             min(candidate_det[dataset_id][1], doc_id),
                                             max(candidate_det[dataset_id][2], doc_id))

            with open(f'.cache_cand_dict.json', 'w') as f:
                json.dump(cand_pool_dict, f)

            with open(f'.cache_cand_count.json', 'w') as f:
                json.dump(candidate_det, f)

        self.cand_pool = cand_pool_dict
        self.candidate_det = candidate_det

    def _get_random_query_prompt(self, dataset_id, query_modality, cand_modality):
        key = f"{dataset_id}, {query_modality}, {cand_modality}"
        prompts = self.query_instructions.get(key, [])
        assert prompts, f"Cannot find prompts for {key}"
        prompt = format_string(random.choice(prompts))
        assert prompt, f"Prompt is empty for {key}"
        return prompt

    def __len__(self):
        if self.is_train:
            return len(self.ds_train)
        else:
            return len(self.ds_validate)

    def get_random_negative_candidate_dids(self, query_dataset_id, modality):
        negative_candidate_dids = []
        tracker = 0
        choice_dataset_ids = [k for k, v in self.candidate_det.items() if v[0] > 0]
        while tracker < self.random_config.CandidateSize:
            if self.random_config.IncludeCrossDomainNegatives:
                neg_dataset_id = random.choice(choice_dataset_ids)
            else:
                neg_dataset_id = query_dataset_id
            doc_id_range = self.candidate_det[neg_dataset_id]
            neg_cand_did = neg_dataset_id + ':' + str(random.randint(doc_id_range[1], doc_id_range[2]))
            neg_cand = self.cand_pool.get(neg_cand_did, None)
            if neg_cand and neg_cand.get("modality") == modality and neg_cand_did not in negative_candidate_dids:
                negative_candidate_dids.append(neg_cand_did)
                tracker += 1

        return negative_candidate_dids

    def get_negative_candidate_dids(self, qid, query_dataset_id, modality):

        if self.random_config.UseModalityNegatives:
            ## TODO::
            ## 1. For the qid, get negatives with incorrect but high ranked modality
            ## 2. For the qid, get negatives with correct but lower ranked than the threshold modality
            pass
        else:
            return self.get_random_negative_candidate_dids(query_dataset_id, modality)

    def _load_and_preprocess_image(self, query_img_path):
        """Load an image given a path"""
        if not query_img_path:
            return None
        image = Image.open(self.config.Common.DataSet.Path + '/' + query_img_path).convert("RGB")
        image = self.feature_processor(images=image, return_tensors='pt')["pixel_values"].squeeze(0)
        return image

    def _prepare_data_dict(self, txt, img_path):
        img = self._load_and_preprocess_image(img_path)
        return {"txt": txt, "img": img}

    def __getitem__(self, idx):
        if self.is_train:
            mbeir_entry = self.ds_train[idx]
        else:
            mbeir_entry = self.ds_validate[idx]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        pos_cand_list = mbeir_entry.get("pos_cand_list", [])

        selected_pos_cand_did = random.choice(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")

        selected_neg_cand_list = []
        if self.is_train:
            selected_neg_cand_id_list = self.get_negative_candidate_dids(qid, query_dataset_id, pos_cand_modality)
            for neg_cand_did in selected_neg_cand_id_list:
                neg_cand = self.cand_pool.get(neg_cand_did)
                neg_cand_txt = neg_cand.get("txt") or ""
                neg_cand_txt = format_string(neg_cand_txt)
                neg_cand["txt"] = neg_cand_txt
                selected_neg_cand_list.append(neg_cand)

        query = self._prepare_data_dict(query_txt_with_prompt, query_img_path)
        instance = {"query": query}

        pos_cand = self._prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
        )
        instance.update({"pos_cand": pos_cand})

        if not self.onlyPrediction:

            neg_cand_list = [
                self._prepare_data_dict(
                    neg_cand["txt"],
                    neg_cand.get("img_path", None),
                )
                for neg_cand in selected_neg_cand_list
            ]
            if len(neg_cand_list) > 0:
                instance.update({"neg_cand_list": neg_cand_list})

        else:
            instance.update({"qid": hash_qid(qid)})
            instance.update({"did": hash_did(selected_pos_cand_did)})
            remaining_pos_cands = []
            remaining_pos_cand_list = pos_cand_list[:]
            remaining_pos_cand_list.remove(selected_pos_cand_did)
            if len(remaining_pos_cand_list) > 0:
                for remaining_pos_cand_did in remaining_pos_cand_list:
                    rem_pos_cand = self.cand_pool.get(remaining_pos_cand_did)
                    rem_pos_cand_txt = rem_pos_cand.get("txt") or ""
                    rem_pos_cand_txt = format_string(rem_pos_cand_txt)
                    rem_pos_cand = self._prepare_data_dict(
                        rem_pos_cand_txt,
                        rem_pos_cand.get("img_path", None),
                    )
                    remaining_pos_cands.append(rem_pos_cand)

            instance.update({"remaining_pos_cand_list": remaining_pos_cands})
            instance.update({"remaining_did": [hash_did(remaining_did) for remaining_did in remaining_pos_cand_list]})

        if not self.is_train:
            instance["modality"] = parse_modality(pos_cand_modality)

        return instance
