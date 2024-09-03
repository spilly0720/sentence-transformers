from __future__ import annotations

import logging
from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
from transformers import AutoModelForMaskedLM

logger = logging.getLogger(__name__)


class MaskedLanguageModelingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer) -> None:
        """
        This loss expects as input pairs of sentences: (masked_sentence, original_sentence).
        During training, the model predicts the masked tokens in the masked_sentence.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
        """
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer

        # Convert the model to a masked language model
        self.mlm = AutoModelForMaskedLM.from_pretrained(
            model[0].auto_model.config._name_or_path
        )
        self.mlm.base_model.load_state_dict(
            model[0].auto_model.state_dict(), strict=False
        )

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        masked_features, target_features = tuple(sentence_features)

        # Get the input_ids and attention_mask for masked and target sentences
        masked_input_ids = masked_features["input_ids"]
        masked_attention_mask = masked_features["attention_mask"]
        target_input_ids = target_features["input_ids"]
        print(masked_input_ids[0])
        print(target_input_ids[0])

        # Forward pass through the MLM
        outputs = self.mlm(
            input_ids=masked_input_ids, attention_mask=masked_attention_mask
        )
        prediction_scores = outputs.logits
        print(torch.argmax(prediction_scores, dim=-1)[0])
        # Create the labels tensor
        mlm_labels = target_input_ids.clone()
        mlm_labels[masked_input_ids != self.tokenizer.mask_token_id] = (
            -100
        )  # We only compute loss on masked tokens

        # Compute the MLM loss
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.tokenizer.vocab_size), mlm_labels.view(-1)
        )

        return masked_lm_loss

    @property
    def citation(self) -> str:
        return """
@article{devlin2018bert,
    title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
    journal={arXiv preprint arXiv:1810.04805},
    year={2018}
}
"""
