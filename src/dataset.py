import torch
from torch.utils.data import Dataset


class AbbreviationDetectionDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_num_dict, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_num_dict = label_to_num_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, example_idx):
        tokens = self.data[example_idx]["tokens"]
        tokenized_input = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = tokenized_input.word_ids(
            batch_index=0
        )  # Map tokens to their respective word.
        previous_word_idx = None
        labels = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                labels.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                labels.append(self.data[example_idx]["labels"][word_idx])
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        labels = list(
            map(lambda x: x if x == -100 else self.label_to_num_dict[x], labels)
        )

        ret = dict(
            input_ids=tokenized_input["input_ids"].squeeze(0),
            attention_mask=tokenized_input["attention_mask"].squeeze(0),
            token_type_ids=tokenized_input["token_type_ids"].squeeze(0),
            labels=torch.tensor(labels),
        )

        return ret
