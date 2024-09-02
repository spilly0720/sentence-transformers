from __future__ import annotations

import random

from sentence_transformers.readers.InputExample import InputExample
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MaskedLanguageModelingDataset(Dataset):
    """
    The MaskedLanguageModelingDataset returns InputExamples in the format: texts=[masked_sentence, original_sentence]
    It is used in combination with the MaskedLanguageModelingLoss: Here, the model tries to predict the masked tokens.

    Args:
        sentences: A list of sentences
        tokenizer: The tokenizer to use for masking
        mlm_probability: Probability of masking a token, default is 0.15
    """

    def __init__(
        self,
        sentences: list[str],
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.15,
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __getitem__(self, item):
        sent = self.sentences[item]
        masked_sent = self.mask_tokens(sent)
        return InputExample(texts=[masked_sent, sent])

    def __len__(self):
        return len(self.sentences)

    def mask_tokens(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)
        masked_tokens = tokens.copy()

        for i in range(len(tokens)):
            if random.random() < self.mlm_probability:
                masked_tokens[i] = self.tokenizer.mask_token

        return self.tokenizer.convert_tokens_to_string(masked_tokens)
