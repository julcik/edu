import math
import random
import re
from collections import Counter
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import nltk

#nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))

def basic_tokenize(line):
    # return re.findall(r"\b\w+\b", line.lower())
    return re.findall(r"\b[a-z]{2,}\b", line.lower()) # get rid of numbers and one letter words


class Word2VecDataset(Dataset):
    def __init__(self, text, window_size=2, mode="skipgram", word2idx=None, num_negative=5, min_count=5, pad_token="<PAD>"):

        self.mode = mode
        self.num_negative = num_negative
        self.window_size = window_size

        if isinstance(text[0], str):
            tokens = []
            for line in text:
                tokens.extend([tok for tok in basic_tokenize(line) if tok not in stop_words])
        else:
            # already tokens
            print("HERE!!")
            tokens = [tok for sent in text for tok in sent]

        #reduce amount of frequent words
        tokens = self.subsample_tokens(tokens, threshold=1e-3)

        if word2idx is None:
            word_freq = Counter(tokens)
            word_freq = {w: c for w, c in word_freq.items() if c >= min_count}

            self.vocab = [pad_token] + sorted(word_freq.keys())
            self.word2idx = {w: i for i, w in enumerate(self.vocab)}
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        else:
            # Use existing vocab (for validation/test)
            self.word2idx = word2idx
            self.idx2word = {i: w for w, i in word2idx.items()}
            self.vocab = list(word2idx.keys())

            # Build frequencies only for tokens present in existing vocab
            word_freq = Counter([t for t in tokens if t in self.word2idx])

        self.pad_idx = self.word2idx.get(pad_token, 0)
        self.vocab_size = len(self.word2idx)
        self.text = [self.word2idx[w] for w in tokens if w in self.word2idx]

        # Negative sampling probas
        freqs = torch.zeros(self.vocab_size)
        freqs[self.pad_idx] = 0
        for w, i in self.word2idx.items():
            freqs[i] = word_freq.get(w, 0)
        self.neg_dist = (freqs ** 0.75)
        self.neg_dist = self.neg_dist / self.neg_dist.sum()

        self.samples = []
        for i in range(len(self.text)):
            window = self.text[max(0, i - window_size): i] + self.text[i + 1: i + 1 + window_size]
            if not window:
                continue
            if mode == "skipgram":
                for w in window:
                    # TODO: just stack?
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
            return {"input": torch.tensor(center, dtype=torch.long), "target": torch.tensor(context, dtype=torch.long),
                "negatives": neg_samples}
        else:
            context, center = self.samples[idx]
            neg_samples = torch.multinomial(self.neg_dist, self.num_negative, replacement=True)
            return {"input": torch.tensor(context, dtype=torch.long), "target": torch.tensor(center, dtype=torch.long),
                "negatives": neg_samples}

    @staticmethod
    def subsample_tokens(tokens, threshold=1e-5):
        counts = Counter(tokens)
        total = len(tokens)
        freqs = {w: c / total for w, c in counts.items()}

        # compute keep probabilities
        keep_probs = {w: min(1.0, math.sqrt(threshold / f) + threshold / f) for w, f in freqs.items()}
        subsampled = [w for w in tokens if random.random() < keep_probs[w]]

        return subsampled

    @staticmethod
    def collate_fn(batch):
        """To handle padding"""
        mode = "skipgram" if batch[0]["input"].dim() == 0 else "cbow"

        if mode == "skipgram":
            return {"input": torch.stack([b["input"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch]), }
        else:
            # Pad variable-length contexts
            inputs = [b["input"] for b in batch]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            return {"input": padded_inputs, "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch]), }

def seed_worker(wid):
    random.seed(torch.initial_seed() % 2**32)

class Word2VecDataModule(pl.LightningDataModule):
    def __init__(self, raw_text, batch_size=128, window_size=2, mode="cbow",
                 num_negative=5, min_count=5, num_workers=4, val_split=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["raw_text"])
        self.raw_text = raw_text

        n_val = int(len(self.raw_text) * self.hparams.val_split)
        train_text = self.raw_text[:-n_val]
        val_text = self.raw_text[-n_val:]

        self.train_dataset = Word2VecDataset(
            train_text,
            window_size=self.hparams.window_size,
            mode=self.hparams.mode,
            num_negative=self.hparams.num_negative,
            min_count=self.hparams.min_count
        )

        # Share vocab with val
        self.val_dataset = Word2VecDataset(
            val_text,
            window_size=self.hparams.window_size,
            mode=self.hparams.mode,
            num_negative=self.hparams.num_negative,
            min_count=self.hparams.min_count,
            word2idx=self.train_dataset.word2idx
        )

        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

        self.vocab_size = self.train_dataset.vocab_size
        self.word2idx = self.train_dataset.word2idx
        self.idx2word = self.train_dataset.idx2word
        self.pad_idx = self.train_dataset.pad_idx

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=Word2VecDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Word2VecDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        # TODO: add test
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Word2VecDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

def debug():
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = dataset["train"]["text"]
    data_module = Word2VecDataModule(
        raw_text=train_text,
        batch_size=16,
        window_size=2,
        mode="skipgram",
        num_negative=10,
        min_count=5
    )
    batch = next(iter(data_module.train_dataloader()))

    batch_size = batch["input"].size(0)

    idx2word = data_module.idx2word

    for i in range(batch_size):
        inpt = idx2word[batch["input"][i].item()]
        target = idx2word[batch["target"][i].item()]
        negatives = [idx2word[neg.item()] for neg in batch["negatives"][i]]

        print(f"{inpt=}, {target=}, {negatives=}")




if __name__ == "__main__":
    debug()