import torch
from torch.utils.data import Dataset
from collections import Counter

class Word2VecDataset(Dataset):
    def __init__(self, text, window_size=2, mode="skipgram", num_negative=5):

        self.mode = mode
        self.num_negative = num_negative

        vocab = sorted(set(text))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)

        self.text = [self.word2idx[w] for w in text]
        self.window_size = window_size

        # Negative sampling
        word_freq = Counter(self.text)
        self.neg_dist = torch.tensor([word_freq[i] ** 0.75 for i in range(self.vocab_size)], dtype=torch.float)
        self.neg_dist /= self.neg_dist.sum()

        self.samples = []
        for i in range(len(self.text)):
            window = self.text[max(0, i - window_size): i] + self.text[i + 1: i + 1 + window_size]
            if not window:
                continue
            if mode == "skipgram":
                for w in window:
                    self.samples.append((self.text[i], w))  # (center, context)
            elif mode == "cbow":
                self.samples.append((window, self.text[i]))  # ([context...], center)
            else:
                raise ValueError("Mode must be 'skipgram' or 'cbow'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "skipgram":
            center, context = self.samples[idx]
            neg_samples = torch.multinomial(self.neg_dist, self.num_negative, replacement=True)
            return {
                "input": torch.tensor(center, dtype=torch.long),
                "target": torch.tensor(context, dtype=torch.long),
                "negatives": neg_samples
            }
        else:
            context, center = self.samples[idx]
            neg_samples = torch.multinomial(self.neg_dist, self.num_negative, replacement=True)
            return {
                "input": torch.tensor(context, dtype=torch.long),
                "target": torch.tensor(center, dtype=torch.long),
                "negatives": neg_samples
            }

    @staticmethod
    def collate_fn(batch):
        """To handle padding"""
        mode = "skipgram" if batch[0]["input"].dim() == 0 else "cbow"

        if mode == "skipgram":
            return {
                "input": torch.stack([b["input"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch]),
            }
        else:
            # Pad variable-length contexts
            inputs = [b["input"] for b in batch]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            return {
                "input": padded_inputs,
                "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch]),
            }
