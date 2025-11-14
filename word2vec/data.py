import math
import random
import re
from collections import Counter, OrderedDict
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from tqdm import tqdm
import hashlib
import pickle
import os
from pathlib import Path

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))


def basic_tokenize(line):
    return re.findall(r"\b[a-z]{2,}\b", line.lower())  # get rid of numbers and one letter words


class Word2VecDataset(Dataset):
    def __init__(self, text, word2idx=None, window_size=2, num_negative=5, min_count=5,
                 pad_token="<PAD>", mode="skipgram", subsample_t=1e-3,
                 cache_dir="./dataset_cache", chunk_size=100, max_cache_chunks=15000):

        self.mode = mode
        self.window_size = window_size
        self.num_negative = num_negative
        self.subsample_t = subsample_t
        self.chunk_size = chunk_size
        self.max_cache_chunks = max_cache_chunks

        # Create cache directory
        self.chunk_dir = Path(cache_dir)
        self.chunk_dir.mkdir(exist_ok=True, parents=True)

        # In-memory chunk cache (LRU-style)
        self.chunk_cache = OrderedDict()  # chunk_id -> chunk_data

        # this is too expensive
        # self.raw_text = text

        # Build vocabulary (same as before)
        print("Building vocabulary...")
        all_tokens = []
        for line in tqdm(text, desc="Tokenizing for vocab"):
            if isinstance(line, str):
                tokens = [tok for tok in basic_tokenize(line) if tok not in stop_words]
            else:
                tokens = line
            all_tokens.extend(tokens)

        word_freq = Counter(all_tokens)
        total = sum(word_freq.values())
        self.keep_prob = {
            w: min(1.0, math.sqrt(subsample_t / (word_freq[w] / total)) + subsample_t / (word_freq[w] / total))
            for w in word_freq}

        if word2idx is None:
            filtered_freq = {w: c for w, c in word_freq.items() if c >= min_count}
            self.vocab = [pad_token] + sorted(filtered_freq.keys())
            self.word2idx = {w: i for i, w in enumerate(self.vocab)}
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        else:
            self.word2idx = word2idx
            self.idx2word = {i: w for w, i in word2idx.items()}
            self.vocab = list(word2idx.keys())
            filtered_freq = {w: word_freq.get(w, 0) for w in self.vocab}

        self.pad_idx = self.word2idx.get(pad_token, 0)
        self.vocab_size = len(self.word2idx)

        # Negative sampling probas
        freqs = torch.zeros(self.vocab_size)
        for w, i in self.word2idx.items():
            freqs[i] = filtered_freq.get(w, 0)
        freqs[self.pad_idx] = 0
        self.neg_dist = (freqs ** 0.75)
        self.neg_dist = self.neg_dist / self.neg_dist.sum()

        # Check if we have cached chunks, if not create them
        self.total_samples = self._load_or_create_chunks(text)
        print(f"Total samples: {self.total_samples}")

    def _load_or_create_chunks(self, raw_text):
        """Load existing chunks or create new ones"""
        meta_file = self.chunk_dir / "meta.pkl"

        if meta_file.exists():
            # Load existing metadata
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            print(f"Loading existing cache with {meta['total_samples']} samples in {meta['num_chunks']} chunks")
            return meta['total_samples']
        else:
            # Create new chunks
            print("Creating new dataset chunks...")
            return self._create_chunks(raw_text)

    def _create_chunks(self, raw_text):
        """Pre-compute all samples and save them in chunks"""
        chunk_samples = []
        total_samples = 0
        chunk_id = 0

        for line_idx, line in enumerate(tqdm(raw_text, desc="Processing lines")):
            line_tokens = self._tokenize_and_subsample(line)

            if len(line_tokens) == 0:
                continue

            # Generate all samples for this line
            line_samples = self._generate_line_samples(line_tokens, line_idx)

            for sample in line_samples:
                chunk_samples.append(sample)
                total_samples += 1

                # Save chunk when it reaches target size
                if len(chunk_samples) >= self.chunk_size:
                    self._save_chunk(chunk_id, chunk_samples)
                    chunk_samples = []
                    chunk_id += 1

        # Save remaining samples
        if chunk_samples:
            self._save_chunk(chunk_id, chunk_samples)
            chunk_id += 1

        # Save metadata
        meta = {
            'total_samples': total_samples,
            'num_chunks': chunk_id,
            'chunk_size': self.chunk_size
        }
        with open(self.chunk_dir / "meta.pkl", 'wb') as f:
            pickle.dump(meta, f)

        print(f"Created {chunk_id} chunks with {total_samples} total samples")
        return total_samples

    def _generate_line_samples(self, token_indices, line_idx):
        """Generate all samples for a line"""
        samples = []

        if self.mode == "skipgram":
            for i in range(len(token_indices)):
                window_start = max(0, i - self.window_size)
                window_end = min(len(token_indices), i + self.window_size + 1)
                for j in range(window_start, window_end):
                    if j != i:
                        center = token_indices[i]
                        context = token_indices[j]
                        samples.append({
                            "input": center,
                            "target": context,
                            "line_idx": line_idx,
                            "sample_type": "skipgram"
                        })
        else:  # cbow
            for center_idx in range(len(token_indices)):
                center = token_indices[center_idx]
                window_start = max(0, center_idx - self.window_size)
                window_end = min(len(token_indices), center_idx + self.window_size + 1)
                context = [token_indices[i] for i in range(window_start, window_end) if i != center_idx]

                if context:  # Only add if we have context
                    samples.append({
                        "input": context,
                        "target": center,
                        "line_idx": line_idx,
                        "sample_type": "cbow"
                    })

        return samples

    def _save_chunk(self, chunk_id, chunk_samples):
        """Save a chunk to disk"""
        chunk_file = self.chunk_dir / f"chunk_{chunk_id}.pkl"
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_samples, f)

    def _load_chunk(self, chunk_id):
        """Load a chunk from disk with caching"""
        # Check if already in cache
        if chunk_id in self.chunk_cache:
            # Move to end (most recently used)
            self.chunk_cache.move_to_end(chunk_id)
            return self.chunk_cache[chunk_id]

        # Load from disk
        chunk_file = self.chunk_dir / f"chunk_{chunk_id}.pkl"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file {chunk_file} not found")

        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)

        # Add to cache
        self.chunk_cache[chunk_id] = chunk_data
        self.chunk_cache.move_to_end(chunk_id)

        # Remove oldest chunks if cache is too large
        while len(self.chunk_cache) > self.max_cache_chunks:
            self.chunk_cache.popitem(last=False)  # Remove oldest (FIFO)

        return chunk_data

    def _tokenize_and_subsample(self, line):
        if isinstance(line, str):
            tokens = [tok for tok in basic_tokenize(line) if tok not in stop_words]
        else:
            tokens = line

        valid_tokens = []
        for pos, tok in enumerate(tokens):
            if tok in self.word2idx:
                if random.random() < self.keep_prob.get(tok, 1.0):
                    valid_tokens.append(self.word2idx[tok])

        return valid_tokens

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """Get sample by index - now with fast disk-based caching!"""
        # Calculate which chunk contains this sample
        chunk_id = idx // self.chunk_size
        sample_offset = idx % self.chunk_size

        # Load chunk (from cache or disk)
        chunk_data = self._load_chunk(chunk_id)
        sample = chunk_data[sample_offset]

        neg_samples = torch.multinomial(self.neg_dist, self.num_negative, replacement=True)

        return {
            "input": torch.tensor(sample["input"], dtype=torch.long),
            "target": torch.tensor(sample["target"], dtype=torch.long),
            "negatives": neg_samples
        }

    def clear_cache(self):
        """Clear disk cache"""
        import shutil
        if self.chunk_dir.exists():
            shutil.rmtree(self.chunk_dir)
            print(f"Cleared cache directory: {self.chunk_dir}")

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0]["input"], torch.Tensor) and batch[0]["input"].dim() == 0:
            # skipgram
            return {
                "input": torch.stack([b["input"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch])
            }
        else:
            # CBOW
            inputs = [b["input"] for b in batch]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            return {
                "input": padded_inputs,
                "target": torch.stack([b["target"] for b in batch]),
                "negatives": torch.stack([b["negatives"] for b in batch])
            }


def seed_worker(wid):
    random.seed(torch.initial_seed() % 2 ** 32)


class Word2VecDataModule(pl.LightningDataModule):
    def __init__(self, raw_text, batch_size=128, window_size=2, mode="cbow",
                 num_negative=5, min_count=5, num_workers=4, val_split=0.1,
                 cache_dir="./dataset_cache", chunk_size=100):
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
            min_count=self.hparams.min_count,
            cache_dir=f"{cache_dir}/train",
            chunk_size=chunk_size
        )

        self.val_dataset = Word2VecDataset(
            val_text,
            window_size=self.hparams.window_size,
            mode=self.hparams.mode,
            num_negative=self.hparams.num_negative,
            min_count=self.hparams.min_count,
            word2idx=self.train_dataset.word2idx,
            cache_dir=f"{cache_dir}/val",
            chunk_size=chunk_size,
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
            persistent_workers=self.hparams.num_workers>0,
            worker_init_fn=seed_worker,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Word2VecDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers>0,
            worker_init_fn=seed_worker,
            pin_memory=True
      )

    def test_dataloader(self):
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
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = dataset["train"]["text"][:1000]  # Use smaller subset for testing

    data_module = Word2VecDataModule(
        raw_text=train_text,
        batch_size=16,
        window_size=5,
        mode="skipgram",
        num_negative=10,
        min_count=5,
        num_workers=0,
    )

    batch = next(iter(data_module.train_dataloader()))
    batch_size = batch["input"].size(0)
    idx2word = data_module.idx2word

    for i in range(min(5, batch_size)):  # Show first 5 samples
        inpt = idx2word[batch["input"][i].item()]
        target = idx2word[batch["target"][i].item()]
        negatives = [idx2word[neg.item()] for neg in batch["negatives"][i]]
        print(f"{inpt=}, {target=}, {negatives=}")


if __name__ == "__main__":
    debug()
    # Word2VecDataset("").clear_cache()