import torch
import numpy as np
import torch.nn.functional as F
from transformers import CLIPProcessor
from datasetUtils.mbier_dataset import MBEIRMainDataset


def _get_padded_text_with_mask(txt):
    return (txt, 1) if txt not in [None, ""] else ("", 0)


def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (torch.zeros((3, 224, 224)), 0)


def get_scores(D, modalities, pred):
    ground_truth = np.arange(D)
    # Find the index with maximum value
    max_candidate_idx = np.argmax(pred, axis=-1)
    # Calculate accuracy based on the selected candidate index
    N = pred.shape[0]
    accuracy = np.sum(np.repeat([ground_truth], N, axis=0) == max_candidate_idx)
    # Calculate modality accuracy based on the selected candidate index
    modality_accuracy = np.sum(modalities == modalities[np.arange(N)[:, None], max_candidate_idx])
    return accuracy, modality_accuracy


class Builder:
    def __init__(self, config, onlyPrediction=False):
        self.config = config
        self.onlyPrediction = onlyPrediction
        if self.onlyPrediction:
            self.bs = self.config.Evaluate.Hyperparameters.EvalBatchSize
        else:
            self.bs = self.config.FineTuning.Hyperparameters.EvalBatchSize

    def get_train_dataset(self):
        return MBEIRMainDataset(self.config)

    def get_eval_dataset(self):
        return MBEIRMainDataset(self.config, is_train=False, onlyPrediction=self.onlyPrediction)

    def get_collate_fn(self):

        Model = self.config.FineTuning.Model
        feature_processor = CLIPProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

        def mbeir_collator(batch):
            txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []

            index_mapping = {
                "query": [[] for _ in range(len(batch))],
            }
            instance_keys = ["query"]

            index_mapping.update({"pos_cand": [[] for _ in range(len(batch))]})
            instance_keys.extend(["pos_cand"])

            qid_list, did_list, remaining_did_list = [], [], []

            if "neg_cand_list" in batch[0]:
                index_mapping.update({"neg_cand_list": [[] for _ in range(len(batch))]})
                instance_keys.extend(["neg_cand_list"])

            if "remaining_pos_cand_list" in batch[0]:
                index_mapping.update({"remaining_pos_cand_list": [[] for _ in range(len(batch))]})
                instance_keys.extend(["remaining_pos_cand_list"])

            if self.onlyPrediction:
                for instance in batch:
                    qid = instance.pop("qid", None)
                    did = instance.pop("did", None)
                    remaining_did = instance.pop("remaining_did", None)
                    if qid is not None:
                        qid_list.append(qid)
                    if did is not None:
                        did_list.append(did)
                    if remaining_did is not None:
                        remaining_did_list.extend(remaining_did)

            # Generate Index Mapping
            counter = 0
            for inst_idx, instance in enumerate(batch):
                for instance_key in instance_keys:
                    items = [instance[instance_key]] if instance_key not in ["neg_cand_list", "remaining_pos_cand_list"] \
                        else instance[instance_key]  # list
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

            tokenized_text = feature_processor(text=txt_list, padding=True, truncation=True, return_tensors='pt')

            txt_batched = tokenized_text['input_ids']
            txt_attention_mask_batched = tokenized_text['attention_mask']

            processed_batch = {
                "txt_batched": txt_batched,
                "txt_attention_mask_batched": txt_attention_mask_batched,
                "image_batched": torch.stack(img_list, dim=0),
                "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
                "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
                "index_mapping": index_mapping,
            }

            if self.onlyPrediction:
                processed_batch.update({"qid_list": torch.tensor(qid_list)})
                processed_batch.update({"did_list": torch.tensor(did_list)})
                if len(remaining_did_list) > 0:
                    processed_batch.update({"remaining_did_list": torch.tensor(remaining_did_list)})

            if "neg_cand_list" not in instance_keys:
                processed_batch['return_loss'] = True
                modalities = torch.tensor([bt["modality"] for bt in batch])
                if modalities.shape[0] < self.bs:
                    expand_dim = self.bs - modalities.shape[0]
                    modalities = F.pad(modalities, pad=(0, expand_dim), mode='constant', value=-2)
                processed_batch['modalities'] = modalities.unsqueeze(0)

            return processed_batch

        return mbeir_collator

    def get_compute(self):

        if self.onlyPrediction:
            bs = self.config.Evaluate.Hyperparameters.EvalBatchSize
        else:
            bs = self.config.FineTuning.Hyperparameters.EvalBatchSize

        def compute(p):

            if self.onlyPrediction:
                predictions = p.predictions[0]
            else:
                predictions = p.predictions

            modalities = p.label_ids

            # To handle last batch predictions
            _, dim = np.where(predictions[-1] == -2.0)
            if len(dim) > 0:
                pred = predictions[:-1,:,:]
                modal = modalities[:-1,:]
                D = int((bs ** 2 - len(dim)) ** 0.5)
            else:
                pred = predictions
                modal = modalities
                D = bs

            accuracy, modality_accuracy = [0,0], [0,0]
            accuracy[0], modality_accuracy[0] = get_scores(bs, modal, pred)

            if D < bs:
                pred = predictions[-1]
                pred = pred[pred > -2.0].reshape(-1, D)[None,:]
                modal = modalities[-1]
                modal = modal[modal > -2.0][None,:]
                accuracy[1], modality_accuracy[1] = get_scores(D, modal, pred)

            T = bs * (predictions.shape[0] - 1) + D
            accuracy = np.sum(accuracy) / T
            modality_accuracy = np.sum(modality_accuracy) / T

            return {
                "accuracy": accuracy,
                "modality_accuracy": modality_accuracy
            }

        return compute
