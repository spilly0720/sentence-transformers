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
        sentences: List[str],
        tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.15,
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __getitem__(self, item):
        sent = self.sentences[item]
        masked_sent = self.mask_whole_words(sent)
        return InputExample(texts=[masked_sent, sent])

    def __len__(self):
        return len(self.sentences)

    def mask_whole_words(self, text: str) -> str:
        words = text.split()
        masked_words = words.copy()

        for i in range(len(words)):
            if random.random() < self.mlm_probability:
                masked_words[i] = self.mask_word(words[i])

        return " ".join(masked_words)

    def mask_word(self, word: str) -> str:
        tokens = self.tokenizer.tokenize(word)
        return " ".join([self.tokenizer.mask_token] * len(tokens))

    @staticmethod
    def get_whole_word_mask(tokens: List[str]) -> List[bool]:
        whole_word_mask = []
        for token in tokens:
            if token.startswith("##"):
                whole_word_mask.append(False)
            else:
                whole_word_mask.append(True)
        return whole_word_mask

    def tokenize_and_mask(self, text: str) -> Tuple[List[str], List[int]]:
        tokens = self.tokenizer.tokenize(text)
        whole_word_mask = self.get_whole_word_mask(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        masked_input_ids = input_ids.copy()
        for i in range(len(tokens)):
            if whole_word_mask[i] and random.random() < self.mlm_probability:
                masked_input_ids[i] = self.tokenizer.mask_token_id
                j = i + 1
                while j < len(tokens) and not whole_word_mask[j]:
                    masked_input_ids[j] = self.tokenizer.mask_token_id
                    j += 1

        return self.tokenizer.convert_ids_to_tokens(masked_input_ids), input_ids
