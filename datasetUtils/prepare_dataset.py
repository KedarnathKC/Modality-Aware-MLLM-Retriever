import torch
import numpy as np
from transformers import CLIPProcessor
from mbier_dataset import MBEIRMainDataset


def _get_padded_text_with_mask(txt):
    return (txt, 1) if txt not in [None, ""] else ("", 0)


def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (torch.zeros((3, 224, 224)), 0)


class Builder:
    def __init__(self, config):
        self.config = config

    def get_train_dataset(self):
        return MBEIRMainDataset(self.config)

    def get_eval_dataset(self):
        return MBEIRMainDataset(self.config, is_train=False)

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

            if "neg_cand_list" in batch[0]:
                index_mapping.update({"neg_cand_list": [[] for _ in range(len(batch))]})
                instance_keys.extend(["neg_cand_list"])

            # Generate Index Mapping
            counter = 0
            for inst_idx, instance in enumerate(batch):
                for instance_key in instance_keys:
                    items = [instance[instance_key]] if instance_key != "neg_cand_list" else instance[
                        instance_key]  # list
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

            if "neg_cand_list" not in instance_keys:
                processed_batch['return_loss'] = True
                processed_batch['modalities'] = torch.tensor([modality["modality"] for modality in batch]).unsqueeze(0)

            return processed_batch

        return mbeir_collator

    def get_compute(self):

        bs = self.config.FineTuning.Hyperparameters.EvalBatchSize

        def compute(p):

            predictions = p.predictions
            modalities = p.label_ids

            ground_truth = np.arange(bs)
            # Find the index with maximum value
            max_candidate_idx = np.argmax(predictions, axis=2)
            # Calculate accuracy based on the selected candidate index
            N = predictions.shape[0]
            accuracy = np.mean(np.repeat([ground_truth], N, axis=0) == max_candidate_idx)

            # Calculate modality accuracy based on the selected candidate index
            modality_accuracy = np.mean(modalities == modalities[np.arange(N)[:, None], max_candidate_idx])

            return {
                "accuracy": accuracy,
                "modality_accuracy": modality_accuracy
            }

        return compute







