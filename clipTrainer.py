from typing import Dict, Union, Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer


class ClipTrainer(Trainer):

    def __init__(self, model, configArgs, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.configArgs = configArgs

        self.loss_function = nn.CrossEntropyLoss()
        self.temperature = self.configArgs.FineTuning.Hyperparameters.Temperature

    def _encode_text(self, text_tensor, txt_attention_mask):
        return self.model.get_text_features(text_tensor, txt_attention_mask)
        # return torch.rand((text_tensor.size()[0], 512), device='mps')

    def _encode_image(self, image_tensor):
        return self.model.get_image_features(image_tensor)
        # return torch.rand((image_tensor.size()[0], 512), device='mps')

    def _fuse_embeddings(self, img_emb, txt_emb):
        fused_emb = img_emb + txt_emb
        return fused_emb

    def _encode_multimodal_input(self, txt_tensor, txt_attention_mask, img_tensor, txt_mask, img_mask):
        """
        :param txt_tensor:
        :param img_tensor:
        :param txt_mask:  expected shape: [batch_size, 1]
        :param img_mask:  expected shape: [batch_size, 1]
        :return:
        """
        txt_emb = self._encode_text(txt_tensor, txt_attention_mask) * txt_mask.unsqueeze(-1)
        img_emb = self._encode_image(img_tensor) * img_mask.unsqueeze(-1)
        return self._fuse_embeddings(txt_emb, img_emb)  # shape: [batch_size, embed_dim]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        txt_batched = inputs["txt_batched"]
        txt_attention_mask_batched = inputs["txt_attention_mask_batched"]
        image_batched = inputs["image_batched"]
        txt_mask_batched = inputs["txt_mask_batched"]
        image_mask_batched = inputs["image_mask_batched"]
        index_mapping = inputs["index_mapping"]
        enable_hard_neg = "neg_cand_list" in index_mapping

        #Compute embeddings
        embeddings = self._encode_multimodal_input(txt_batched, txt_attention_mask_batched,
                                                   image_batched, txt_mask_batched, image_mask_batched)

        #Extract embeddings
        q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]  # shape: [bs, embed_dim]
        p_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]  # shape: [bs, embed_dim]
        n_embeds = None
        if enable_hard_neg:
            n_embeds = embeddings[torch.tensor(index_mapping["neg_cand_list"])]  # [bs, neg_num, embed_dim]

        # Normalized features
        q_embeds = F.normalize(q_embeds, dim=-1)
        p_embeds = F.normalize(p_embeds, dim=-1)

        if enable_hard_neg:
            positive_logit = torch.sum(q_embeds * p_embeds, dim=1, keepdim=True)

            query = q_embeds.unsqueeze(1)
            negative_logits = query @ n_embeds.transpose(-2, -1)
            negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        else:
            logits = q_embeds @ p_embeds.transpose(-2, -1)
            labels = torch.arange(len(q_embeds), device=q_embeds.device)

        loss = self.loss_function(logits / self.temperature, labels)

        return (loss, logits) if return_outputs else loss