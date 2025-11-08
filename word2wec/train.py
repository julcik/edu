import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from collections import Counter
import random

from word2wec.data import Word2VecDataset
from word2wec.model import Word2Vec, Word2VecRunner

# TODO: this is for debug only
text = "we all live in a yellow submarine we all live in the sea".split()

def train(mode = "cbow"):
    dataset = Word2VecDataset(text, window_size=2, mode=mode, num_negative=5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=Word2VecDataset.collate_fn)

    runner = Word2VecRunner(dataset.vocab_size, embedding_dim=50, mode=mode)
    trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False, logger=False)
    trainer.fit(runner, loader)

if __name__ == "__main__":
    train()
